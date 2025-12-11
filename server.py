"""
Dia2 Streaming TTS Server with Persistent State

Designed for single-conversation, low-latency voice call applications.
Keeps model state warm between turns to minimize latency after initial warmup.

Usage:
    python server.py
    
Then connect via WebSocket to ws://localhost:3030
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, AsyncIterator
from collections import deque

import torch
import numpy as np

# Local vendored dia2
from dia2.engine import Dia2
from dia2.generation import GenerationConfig, SamplingConfig, merge_generation_config, normalize_script
from dia2.runtime.context import RuntimeContext
from dia2.runtime.script_parser import parse_script
from dia2.runtime.generator import build_initial_state, GenerationState
from dia2.runtime.state_machine import StateMachine, State, Entry
from dia2.runtime.guidance import apply_classifier_guidance, sample_audio_logits
from dia2.runtime.sampler import sample_token
from dia2.audio.grid import mask_audio_logits, undelay_frames
from dia2.runtime.logger import RuntimeLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """A chunk of generated audio."""
    samples: np.ndarray  # PCM float32, -1 to 1
    sample_rate: int
    frame_start: int
    frame_end: int
    is_final: bool = False


@dataclass 
class StreamingSession:
    """
    Persistent session state for a single conversation.
    
    Keeps all model state warm between turns to minimize latency.
    """
    runtime: RuntimeContext
    gen_state: GenerationState
    sm_state: State
    config: GenerationConfig
    
    # Generation tracking
    current_step: int = 0
    last_decoded_step: int = -1  # Last step we decoded up to
    last_sent_sample: int = 0    # Last PCM sample index we sent
    first_word_frame: Optional[int] = None
    eos_cutoff: Optional[int] = None
    
    # Delay info (cached from runtime)
    max_delay: int = 0
    flush_tail: int = 0
    
    # Pre-allocated buffers
    positions: torch.Tensor = None
    main_tokens_buf: torch.Tensor = None
    aux_tokens_buf: torch.Tensor = None
    text_logits_buf: torch.Tensor = None
    cb0_logits_buf: torch.Tensor = None
    dep_logits_buf: List[torch.Tensor] = field(default_factory=list)
    
    # State flags
    is_warmed_up: bool = False
    is_generating: bool = False
    generation_complete: bool = False
    
    # Timestamps for current generation
    accumulated_timestamps: List[Tuple[str, int]] = field(default_factory=list)


class Dia2StreamingServer:
    """
    Streaming TTS server optimized for single-conversation low-latency.
    """
    
    def __init__(
        self,
        model_repo: str = "nari-labs/Dia2-2B",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        self.model_repo = model_repo
        self.device = device
        self.dtype = dtype
        
        self.dia: Optional[Dia2] = None
        self.session: Optional[StreamingSession] = None
        
        # Configuration
        self.default_config = GenerationConfig(
            cfg_scale=3.0,
            text=SamplingConfig(temperature=0.7, top_k=50),
            audio=SamplingConfig(temperature=0.8, top_k=80),
            initial_padding=0,  # Minimize for low latency
            use_cuda_graph=False,
            use_torch_compile=False,
        )
        
        # Streaming settings - decode every N frames
        self.decode_every_n_frames = 1
        
    async def initialize(self):
        """Load the model. Call once at startup."""
        logger.info(f"Loading Dia2 model from {self.model_repo}...")
        start = time.time()
        
        self.dia = Dia2.from_repo(
            self.model_repo,
            device=self.device,
            dtype=self.dtype,
        )
        self.dia.default_config = self.default_config
        
        # Force runtime initialization
        _ = self.dia._ensure_runtime()
        
        logger.info(f"Model loaded in {time.time() - start:.2f}s")
        logger.info(f"Sample rate: {self.dia.sample_rate}")
        logger.info(f"Max context steps: {self.dia.max_context_steps}")
        
    def _create_session(self, initial_text: str) -> StreamingSession:
        """Create a new streaming session with initial text."""
        runtime = self.dia._ensure_runtime()
        config = self.default_config
        
        # Parse initial script
        text = normalize_script(initial_text)
        entries = list(parse_script([text], runtime.tokenizer, runtime.constants, runtime.frame_rate))
        
        # Initialize state machine
        runtime.machine.initial_padding = config.initial_padding
        sm_state = runtime.machine.new_state(entries)
        
        # Build generation state
        gen_state = build_initial_state(runtime, prefix=None)
        
        # Calculate delays
        delay_tensor = runtime.audio_delay_tensor
        max_delay = int(delay_tensor.max().item()) if delay_tensor.numel() else 0
        flush_tail = max_delay + getattr(runtime.machine, "max_padding", 0)
        
        # Pre-allocate buffers
        branches = gen_state.step_tokens.shape[0]
        device = runtime.device
        data_cfg = runtime.config.data
        logits_dtype = runtime.precision.logits
        
        positions = torch.empty(1, 1, dtype=torch.long, device=device)
        main_tokens_buf = torch.empty(branches, dtype=torch.long, device=device)
        aux_tokens_buf = torch.empty(branches, dtype=torch.long, device=device)
        text_logits_buf = torch.empty((branches, 1, data_cfg.action_vocab_size), dtype=logits_dtype, device=device)
        cb0_logits_buf = torch.empty((branches, 1, data_cfg.audio_vocab_size), dtype=logits_dtype, device=device)
        
        dep_vocab = runtime.model.depformer.audio_vocab_limit or data_cfg.audio_vocab_size
        dep_logits_buf = [
            torch.empty((branches, 1, 1, dep_vocab), dtype=logits_dtype, device=device)
            for _ in range(runtime.model.depformer.num_depth)
        ]
        
        session = StreamingSession(
            runtime=runtime,
            gen_state=gen_state,
            sm_state=sm_state,
            config=config,
            current_step=0,
            last_decoded_step=-1,
            last_sent_sample=0,
            max_delay=max_delay,
            flush_tail=flush_tail,
            positions=positions,
            main_tokens_buf=main_tokens_buf,
            aux_tokens_buf=aux_tokens_buf,
            text_logits_buf=text_logits_buf,
            cb0_logits_buf=cb0_logits_buf,
            dep_logits_buf=dep_logits_buf,
        )
        
        logger.info(f"Session created: max_delay={max_delay} frames ({max_delay * 80}ms)")
        return session
    
    def inject_text(self, text: str):
        """Inject new text into the current session."""
        if self.session is None:
            raise RuntimeError("No active session. Call start_session() first.")
        
        runtime = self.session.runtime
        text = normalize_script(text)
        new_entries = list(parse_script([text], runtime.tokenizer, runtime.constants, runtime.frame_rate))
        
        # Append to existing entries
        self.session.sm_state.entries.extend(new_entries)
        
        # Reset EOS tracking since we have more content
        self.session.eos_cutoff = None
        self.session.sm_state.end_step = None
        self.session.generation_complete = False
        
        logger.info(f"Injected {len(new_entries)} entries, queue size: {len(self.session.sm_state.entries)}")
    
    async def start_session(self, initial_text: str) -> AsyncIterator[AudioChunk]:
        """Start a new session with initial text and begin streaming."""
        self.session = self._create_session(initial_text)
        self.session.is_generating = True
        
        async for chunk in self._generation_loop():
            yield chunk
    
    async def continue_generation(self) -> AsyncIterator[AudioChunk]:
        """Continue generating from current state after inject_text()."""
        if self.session is None:
            raise RuntimeError("No active session")
        
        self.session.is_generating = True
        self.session.generation_complete = False
        
        async for chunk in self._generation_loop():
            yield chunk
    
    def _decode_and_get_new_samples(
        self,
        up_to_step: int,
        is_final: bool = False
    ) -> Optional[AudioChunk]:
        """
        Decode audio buffer up to the given step and return only NEW samples.
        """
        session = self.session
        runtime = session.runtime
        audio_buf = session.gen_state.audio_buf
        token_ids = runtime.constants
        
        # We need at least max_delay+1 steps to get any aligned frames
        if up_to_step < session.max_delay:
            return None
        
        # Extract tokens from start to up_to_step
        chunk_tokens = audio_buf[0, :, :up_to_step + 1].clone()
        
        # Replace ungenerated with pad
        chunk_tokens = torch.where(
            chunk_tokens == token_ids.ungenerated,
            torch.full_like(chunk_tokens, token_ids.audio_pad),
            chunk_tokens
        )
        
        # Undelay the full buffer
        aligned = undelay_frames(chunk_tokens, runtime.audio_delays, token_ids.audio_pad)
        
        if aligned.shape[-1] == 0:
            return None
        
        # Decode to PCM
        aligned = aligned.unsqueeze(0)  # Add batch dim
        with torch.inference_mode():
            pcm = runtime.mimi.decode(aligned)
            pcm = pcm[0, 0]  # Remove batch and channel
        
        total_samples = pcm.shape[-1]
        
        # Only return samples we haven't sent yet
        if session.last_sent_sample >= total_samples:
            return None
        
        new_samples = pcm[session.last_sent_sample:].cpu().numpy().astype(np.float32)
        
        if len(new_samples) == 0:
            return None
        
        start_sample = session.last_sent_sample
        session.last_sent_sample = total_samples
        session.last_decoded_step = up_to_step
        
        logger.info(f"Decoded step {up_to_step}: {len(new_samples)} new samples")
        
        return AudioChunk(
            samples=new_samples,
            sample_rate=runtime.mimi.sample_rate,
            frame_start=start_sample,
            frame_end=total_samples,
            is_final=is_final,
        )
    
    async def _generation_loop(self) -> AsyncIterator[AudioChunk]:
        """Main generation loop. Yields audio chunks as frames become available."""
        session = self.session
        runtime = session.runtime
        gen_state = session.gen_state
        sm_state = session.sm_state
        config = session.config
        
        step_tokens = gen_state.step_tokens
        audio_buf = gen_state.audio_buf
        branches = step_tokens.shape[0]
        token_ids = runtime.constants
        delay_tensor = runtime.audio_delay_tensor
        
        cfg_active = config.cfg_scale != 1.0
        positions_view = session.positions.expand(branches, -1)
        
        max_context = runtime.config.runtime.max_context_steps
        
        frames_since_last_decode = 0
        
        logger.info(f"Starting generation loop at step {session.current_step}")
        
        with torch.inference_mode():
            while session.is_generating:
                t = session.current_step
                
                # Check termination
                if session.eos_cutoff is not None and t >= session.eos_cutoff:
                    session.generation_complete = True
                    break
                if t + 1 >= audio_buf.shape[-1]:
                    logger.warning("Reached max context length")
                    break
                if t >= max_context:
                    logger.warning("Reached max context steps")
                    break
                
                # Reset depformer cache
                gen_state.decode.depformer.reset()
                session.positions.fill_(t)
                
                # Fill audio channels with delayed tokens
                num_audio_channels = delay_tensor.numel()
                if num_audio_channels > 0:
                    target = step_tokens[:, 2:2 + num_audio_channels, 0]
                    if t < audio_buf.shape[-1]:
                        target.copy_(audio_buf[:, :num_audio_channels, t])
                    else:
                        target.fill_(token_ids.audio_bos)
                    mask = delay_tensor > t
                    mask_expanded = mask.unsqueeze(0).expand_as(target)
                    target.copy_(torch.where(mask_expanded, token_ids.audio_bos, target))
                
                # Set unconditional branch for CFG
                if branches > 1:
                    step_tokens[1:, 0, 0] = token_ids.zero
                    step_tokens[1:, 1, 0] = token_ids.pad
                
                # Transformer forward
                hidden_t, text_logits_t, cb0_logits_t, present = runtime.transformer_step(
                    step_tokens, positions_view, gen_state.decode.transformer
                )
                session.text_logits_buf.copy_(text_logits_t)
                session.cb0_logits_buf.copy_(cb0_logits_t)
                gen_state.decode.transformer = present
                
                # Sample text token
                guided_text = apply_classifier_guidance(
                    session.text_logits_buf, cfg_active, config.cfg_scale, config.cfg_filter_k
                )
                if guided_text.shape[0] > 1:
                    guided_text = guided_text[:1]
                text_token = sample_token(
                    guided_text, temp=config.text.temperature, top_k=config.text.top_k
                ).item()
                
                # Process state machine
                main_token, aux_token, _ = runtime.machine.process(t, sm_state, text_token)
                second_token = aux_token if aux_token != -1 else token_ids.pad
                
                # Track first word
                if session.first_word_frame is None and main_token == token_ids.new_word:
                    session.first_word_frame = t - config.initial_padding
                
                step_tokens[:, 0, 0] = main_token
                step_tokens[:, 1, 0] = second_token
                
                # Sample cb0
                guided_cb0 = apply_classifier_guidance(
                    session.cb0_logits_buf, cfg_active, config.cfg_scale, config.cfg_filter_k
                )
                if guided_cb0.shape[0] > 1:
                    guided_cb0 = guided_cb0[:1]
                masked_cb0 = mask_audio_logits(guided_cb0, token_ids.audio_pad, token_ids.audio_bos)
                codebook_token = sample_audio_logits(masked_cb0, config.audio.temperature, config.audio.top_k)
                audio_buf[:, 0, t + 1] = codebook_token
                
                # Depformer stages
                prev_audio = codebook_token.expand(branches)
                session.main_tokens_buf.fill_(main_token)
                session.aux_tokens_buf.fill_(second_token)
                
                for stage in range(runtime.model.depformer.num_depth):
                    logits_stage, dep_present = runtime.depformer_step(
                        prev_audio=prev_audio,
                        transformer_out=hidden_t,
                        stage_index=stage,
                        cache=gen_state.decode.depformer,
                        main_text=session.main_tokens_buf if stage == 0 else None,
                        second_text=session.aux_tokens_buf if stage == 0 else None,
                    )
                    session.dep_logits_buf[stage].copy_(logits_stage)
                    gen_state.decode.depformer = dep_present
                    
                    dep_logits = apply_classifier_guidance(
                        session.dep_logits_buf[stage], cfg_active, config.cfg_scale, config.cfg_filter_k
                    )
                    if dep_logits.shape[0] > 1:
                        dep_logits = dep_logits[:1]
                    stage_token = sample_audio_logits(
                        dep_logits, config.audio.temperature, config.audio.top_k
                    )
                    audio_buf[:, stage + 1, t + 1] = stage_token
                    prev_audio = stage_token.expand(branches)
                
                # Check for EOS
                if session.eos_cutoff is None and sm_state.end_step is not None:
                    session.eos_cutoff = sm_state.end_step + session.flush_tail
                
                session.current_step = t + 1
                frames_since_last_decode += 1
                
                # Check if we should decode and yield
                can_decode = (t + 1) > session.max_delay
                should_decode = can_decode and frames_since_last_decode >= self.decode_every_n_frames
                
                is_final = (session.eos_cutoff is not None and t + 1 >= session.eos_cutoff)
                
                if is_final:
                    should_decode = True
                
                if should_decode:
                    chunk = self._decode_and_get_new_samples(t + 1, is_final)
                    if chunk is not None:
                        frames_since_last_decode = 0
                        yield chunk
                
                # Yield control for async every few steps
                if t % 4 == 0:
                    await asyncio.sleep(0)
                
                if is_final:
                    break
        
        session.is_generating = False
        logger.info(f"Generation loop ended at step {session.current_step}")
        
        # Final flush
        if not session.generation_complete:
            chunk = self._decode_and_get_new_samples(session.current_step, is_final=True)
            if chunk is not None:
                yield chunk
    
    def pause_generation(self):
        """Pause generation but keep state warm."""
        if self.session:
            self.session.is_generating = False
    
    def reset_session(self):
        """Fully reset the session."""
        self.session = None


# ============================================================================
# WebSocket Server
# ============================================================================

async def handle_websocket(websocket, server: Dia2StreamingServer):
    """Handle a WebSocket connection for streaming TTS."""
    import struct
    
    logger.info("Client connected")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                cmd = data.get("cmd")
                
                if cmd == "start":
                    text = data.get("text", "")
                    logger.info(f"Starting session with: {text[:50]}...")
                    
                    async for chunk in server.start_session(text):
                        pcm16 = (chunk.samples * 32767).astype(np.int16)
                        header = struct.pack("<I", len(pcm16))
                        await websocket.send(header + pcm16.tobytes())
                        
                        if chunk.is_final:
                            await websocket.send(json.dumps({"event": "done"}))
                
                elif cmd == "inject":
                    text = data.get("text", "")
                    logger.info(f"Injecting: {text[:50]}...")
                    server.inject_text(text)
                    
                    async for chunk in server.continue_generation():
                        pcm16 = (chunk.samples * 32767).astype(np.int16)
                        header = struct.pack("<I", len(pcm16))
                        await websocket.send(header + pcm16.tobytes())
                        
                        if chunk.is_final:
                            await websocket.send(json.dumps({"event": "done"}))
                
                elif cmd == "pause":
                    server.pause_generation()
                    await websocket.send(json.dumps({"event": "paused"}))
                
                elif cmd == "reset":
                    server.reset_session()
                    await websocket.send(json.dumps({"event": "reset"}))
                
                else:
                    await websocket.send(json.dumps({"error": f"Unknown command: {cmd}"}))
                    
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"error": "Invalid JSON"}))
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Client disconnected")


async def main():
    import websockets
    
    server = Dia2StreamingServer(
        model_repo="nari-labs/Dia2-2B",
        device="cuda",
        dtype="bfloat16",
    )
    
    await server.initialize()
    
    logger.info("Starting WebSocket server on ws://0.0.0.0:3030")
    
    async with websockets.serve(
        lambda ws: handle_websocket(ws, server),
        "0.0.0.0",
        3030,
    ):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
