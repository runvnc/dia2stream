"""
Dia2 Streaming TTS Server with Persistent State and Voice Cloning

Usage:
    python server.py --seed 42
    
Then connect via WebSocket to ws://localhost:3030
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, AsyncIterator
from pathlib import Path

import torch
import numpy as np

from dia2.engine import Dia2
from dia2.generation import GenerationConfig, SamplingConfig, PrefixConfig, normalize_script
from dia2.runtime.context import RuntimeContext
from dia2.runtime.script_parser import parse_script
from dia2.runtime.generator import build_initial_state, GenerationState, warmup_with_prefix
from dia2.runtime.state_machine import State, Entry
from dia2.runtime.guidance import apply_classifier_guidance, sample_audio_logits
from dia2.runtime.sampler import sample_token
from dia2.audio.grid import mask_audio_logits, undelay_frames
from dia2.runtime.voice_clone import build_prefix_plan, PrefixPlan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    samples: np.ndarray
    sample_rate: int
    frame_start: int
    frame_end: int
    is_final: bool = False
    latency_ms: Optional[float] = None


@dataclass 
class StreamingSession:
    runtime: RuntimeContext
    gen_state: GenerationState
    sm_state: State
    config: GenerationConfig
    prefix_plan: Optional[PrefixPlan] = None
    
    current_step: int = 0
    last_sent_sample: int = 0
    first_word_frame: Optional[int] = None
    eos_cutoff: Optional[int] = None
    
    generation_start_step: int = 0
    segment_start_step: int = 0
    segment_samples_sent: int = 0
    
    # For inject: decode from earlier in buffer to get instant audio
    decode_start_step: int = 0
    
    # Incremental decode tracking
    last_decode_step: int = 0
    decoded_pcm_cache: Optional[np.ndarray] = None
    
    request_start_time: float = 0.0
    first_chunk_sent: bool = False
    
    max_delay: int = 0
    flush_tail: int = 0
    
    positions: torch.Tensor = None
    main_tokens_buf: torch.Tensor = None
    aux_tokens_buf: torch.Tensor = None
    text_logits_buf: torch.Tensor = None
    cb0_logits_buf: torch.Tensor = None
    dep_logits_buf: List[torch.Tensor] = field(default_factory=list)
    
    is_generating: bool = False
    generation_complete: bool = False


class Dia2StreamingServer:
    def __init__(
        self,
        model_repo: str = "nari-labs/Dia2-2B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        prefix_speaker_1: Optional[str] = None,
        prefix_speaker_2: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.model_repo = model_repo
        self.device = device
        self.dtype = dtype
        self.prefix_speaker_1 = prefix_speaker_1
        self.prefix_speaker_2 = prefix_speaker_2
        self.seed = seed
        
        self.dia: Optional[Dia2] = None
        self.session: Optional[StreamingSession] = None
        self._cached_prefix_plan: Optional[PrefixPlan] = None
        
        self.default_config = GenerationConfig(
            cfg_scale=3.0,
            text=SamplingConfig(temperature=0.7, top_k=50),
            audio=SamplingConfig(temperature=0.8, top_k=80),
            initial_padding=0,
            use_cuda_graph=False,
            use_torch_compile=False,
        )
        
        # Decode every N frames to reduce overhead
        # Lower = more responsive but more decode calls
        # Higher = fewer decode calls but chunkier audio
        self.decode_every_n_frames = 3
        
    def _set_seed(self):
        """Set random seed for reproducible generation."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            np.random.seed(self.seed)
            logger.info(f"Random seed set to {self.seed}")
        
    async def initialize(self):
        logger.info(f"Loading Dia2 model from {self.model_repo}...")
        start = time.time()
        
        self.dia = Dia2.from_repo(
            self.model_repo,
            device=self.device,
            dtype=self.dtype,
        )
        self.dia.default_config = self.default_config
        runtime = self.dia._ensure_runtime()
        
        logger.info(f"Model loaded in {time.time() - start:.2f}s")
        logger.info(f"Sample rate: {self.dia.sample_rate}")
        
        # Warmup
        self._set_seed()
        await self._warmup()
        
    async def _warmup(self):
        logger.info("Running warmup...")
        warmup_start = time.time()
        
        chunks = []
        async for chunk in self.start_session("[S1] Hello."):
            chunks.append(chunk)
        
        self.session = None
        logger.info(f"Warmup done in {time.time() - warmup_start:.2f}s")
        logger.info("Server ready!")
        
    def _create_session(self, initial_text: str, request_time: float) -> StreamingSession:
        self._set_seed()  # Reset seed for reproducible generation
        
        runtime = self.dia._ensure_runtime()
        config = self.default_config
        
        text = normalize_script(initial_text)
        entries = list(parse_script([text], runtime.tokenizer, runtime.constants, runtime.frame_rate))
        
        runtime.machine.initial_padding = config.initial_padding
        sm_state = runtime.machine.new_state(entries)
        gen_state = build_initial_state(runtime, prefix=None)
        
        delay_tensor = runtime.audio_delay_tensor
        max_delay = int(delay_tensor.max().item()) if delay_tensor.numel() else 0
        flush_tail = max_delay + getattr(runtime.machine, "max_padding", 0)
        
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
            prefix_plan=None,
            current_step=0,
            generation_start_step=0,
            segment_start_step=0,
            segment_samples_sent=0,
            decode_start_step=0,
            last_decode_step=0,
            decoded_pcm_cache=None,
            last_sent_sample=0,
            request_start_time=request_time,
            max_delay=max_delay,
            flush_tail=flush_tail,
            positions=positions,
            main_tokens_buf=main_tokens_buf,
            aux_tokens_buf=aux_tokens_buf,
            text_logits_buf=text_logits_buf,
            cb0_logits_buf=cb0_logits_buf,
            dep_logits_buf=dep_logits_buf,
        )
        
        setup_time = (time.time() - request_time) * 1000
        logger.info(f"Session created in {setup_time:.0f}ms")
        return session
    
    def inject_text(self, text: str, request_time: float):
        if self.session is None:
            raise RuntimeError("No active session")
        
        session = self.session
        runtime = session.runtime
        
        # For inject, we can decode from earlier in the buffer to get audio faster
        # Instead of waiting for max_delay NEW frames, use existing buffer
        # decode_start_step = max(0, current_step - max_delay) gives us instant decode capability
        decode_start = max(0, session.current_step - session.max_delay)
        
        session.decode_start_step = decode_start
        session.segment_start_step = decode_start  # Decode from earlier point
        session.segment_samples_sent = 0
        
        # But track where new content starts for proper audio slicing
        session.last_decode_step = session.current_step
        session.decoded_pcm_cache = None
        
        text = normalize_script(text)
        new_entries = list(parse_script([text], runtime.tokenizer, runtime.constants, runtime.frame_rate))
        
        session.sm_state.entries.extend(new_entries)
        session.eos_cutoff = None
        session.sm_state.end_step = None
        session.generation_complete = False
        session.request_start_time = request_time
        session.first_chunk_sent = False
        
        logger.info(f"Injected {len(new_entries)} entries at step {session.segment_start_step}")
    
    async def start_session(self, initial_text: str) -> AsyncIterator[AudioChunk]:
        request_time = time.time()
        self.session = self._create_session(initial_text, request_time)
        self.session.is_generating = True
        
        async for chunk in self._generation_loop():
            yield chunk
    
    async def continue_generation(self) -> AsyncIterator[AudioChunk]:
        if self.session is None:
            raise RuntimeError("No active session")
        
        self.session.is_generating = True
        self.session.generation_complete = False
        
        async for chunk in self._generation_loop():
            yield chunk
    
    def _decode_incremental(self, up_to_step: int, is_final: bool = False, is_inject: bool = False) -> Optional[AudioChunk]:
        """Decode audio frames. For inject, can use existing buffer for instant audio."""
        session = self.session
        runtime = session.runtime
        audio_buf = session.gen_state.audio_buf
        token_ids = runtime.constants
        
        # Calculate frames available for decoding
        # For inject: decode_start_step is set earlier in buffer, so we have frames immediately
        # For start: decode_start_step = segment_start_step = 0
        decode_start = session.decode_start_step
        total_frames = up_to_step - decode_start
        
        if total_frames <= session.max_delay:
            return None  # Not enough frames yet
        
        # Calculate how many NEW aligned frames we can decode
        # We need to decode from a point that gives us new aligned frames
        # The aligned output length is: total_frames - max_delay
        
        # For incremental decode, we decode the full range but only keep new samples
        # This is because Mimi needs context for proper decoding
        chunk_tokens = audio_buf[0, :, decode_start:up_to_step].clone()
        
        # Replace ungenerated tokens with pad
        chunk_tokens = torch.where(
            chunk_tokens == token_ids.ungenerated,
            torch.full_like(chunk_tokens, token_ids.audio_pad),
            chunk_tokens
        )
        
        # Undelay to align codebooks
        aligned = undelay_frames(chunk_tokens, runtime.audio_delays, token_ids.audio_pad)
        
        if aligned.shape[-1] == 0:
            return None
        
        # Decode to PCM
        aligned = aligned.unsqueeze(0)
        with torch.inference_mode():
            pcm = runtime.mimi.decode(aligned)
            pcm = pcm[0, 0]
        
        # Convert to numpy
        pcm_np = pcm.cpu().numpy().astype(np.float32)
        total_samples = len(pcm_np)
        
        # Only send samples we haven't sent yet
        if session.segment_samples_sent >= total_samples:
            return None
        
        new_samples = pcm_np[session.segment_samples_sent:]
        
        if len(new_samples) == 0:
            return None
        
        # Calculate latency
        now = time.time()
        latency_ms = (now - session.request_start_time) * 1000
        
        if not session.first_chunk_sent:
            session.first_chunk_sent = True
            logger.info(f"*** FIRST CHUNK LATENCY: {latency_ms:.0f}ms ***")
        
        start_sample = session.segment_samples_sent
        session.segment_samples_sent = total_samples
        session.last_decode_step = up_to_step
        
        return AudioChunk(
            samples=new_samples,
            sample_rate=runtime.mimi.sample_rate,
            frame_start=start_sample,
            frame_end=total_samples,
            is_final=is_final,
            latency_ms=latency_ms,
        )
    
    async def _generation_loop(self) -> AsyncIterator[AudioChunk]:
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
        
        with torch.inference_mode():
            while session.is_generating:
                t = session.current_step
                
                if session.eos_cutoff is not None and t >= session.eos_cutoff:
                    session.generation_complete = True
                    break
                if t + 1 >= audio_buf.shape[-1] or t >= max_context:
                    break
                
                gen_state.decode.depformer.reset()
                session.positions.fill_(t)
                
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
                
                if branches > 1:
                    step_tokens[1:, 0, 0] = token_ids.zero
                    step_tokens[1:, 1, 0] = token_ids.pad
                
                hidden_t, text_logits_t, cb0_logits_t, present = runtime.transformer_step(
                    step_tokens, positions_view, gen_state.decode.transformer
                )
                session.text_logits_buf.copy_(text_logits_t)
                session.cb0_logits_buf.copy_(cb0_logits_t)
                gen_state.decode.transformer = present
                
                guided_text = apply_classifier_guidance(
                    session.text_logits_buf, cfg_active, config.cfg_scale, config.cfg_filter_k
                )
                if guided_text.shape[0] > 1:
                    guided_text = guided_text[:1]
                text_token = sample_token(
                    guided_text, temp=config.text.temperature, top_k=config.text.top_k
                ).item()
                
                main_token, aux_token, _ = runtime.machine.process(t, sm_state, text_token)
                second_token = aux_token if aux_token != -1 else token_ids.pad
                
                if session.first_word_frame is None and main_token == token_ids.new_word:
                    session.first_word_frame = t - config.initial_padding
                
                step_tokens[:, 0, 0] = main_token
                step_tokens[:, 1, 0] = second_token
                
                guided_cb0 = apply_classifier_guidance(
                    session.cb0_logits_buf, cfg_active, config.cfg_scale, config.cfg_filter_k
                )
                if guided_cb0.shape[0] > 1:
                    guided_cb0 = guided_cb0[:1]
                masked_cb0 = mask_audio_logits(guided_cb0, token_ids.audio_pad, token_ids.audio_bos)
                codebook_token = sample_audio_logits(masked_cb0, config.audio.temperature, config.audio.top_k)
                audio_buf[:, 0, t + 1] = codebook_token
                
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
                
                if session.eos_cutoff is None and sm_state.end_step is not None:
                    session.eos_cutoff = sm_state.end_step + session.flush_tail
                
                session.current_step = t + 1
                frames_since_last_decode += 1
                
                frames_in_segment = (t + 1) - session.segment_start_step
                frames_for_decode = (t + 1) - session.decode_start_step
                can_decode = frames_for_decode > session.max_delay
                should_decode = can_decode and frames_since_last_decode >= self.decode_every_n_frames

                is_final = (session.eos_cutoff is not None and t + 1 >= session.eos_cutoff)
                if is_final:
                    should_decode = True
                
                if should_decode:
                    chunk = self._decode_incremental(t + 1, is_final)
                    if chunk is not None:
                        frames_since_last_decode = 0
                        yield chunk
                
                if t % 4 == 0:
                    await asyncio.sleep(0)
                
                if is_final:
                    break
        
        session.is_generating = False
        
        if not session.generation_complete:
            chunk = self._decode_incremental(session.current_step, is_final=True)
            if chunk is not None:
                yield chunk
    
    def pause_generation(self):
        if self.session:
            self.session.is_generating = False
    
    def reset_session(self):
        self.session = None


async def handle_websocket(websocket, server: Dia2StreamingServer):
    import struct
    
    logger.info("Client connected")
    
    try:
        async for message in websocket:
            try:
                request_time = time.time()
                data = json.loads(message)
                cmd = data.get("cmd")
                
                if cmd == "start":
                    text = data.get("text", "")
                    logger.info(f"START: {text[:50]}...")
                    
                    async for chunk in server.start_session(text):
                        pcm16 = (chunk.samples * 32767).astype(np.int16)
                        header = struct.pack("<If", len(pcm16), chunk.latency_ms or 0)
                        await websocket.send(header + pcm16.tobytes())
                        if chunk.is_final:
                            await websocket.send(json.dumps({"event": "done"}))
                
                elif cmd == "inject":
                    text = data.get("text", "")
                    logger.info(f"INJECT: {text[:50]}...")
                    server.inject_text(text, request_time)
                    
                    async for chunk in server.continue_generation():
                        pcm16 = (chunk.samples * 32767).astype(np.int16)
                        header = struct.pack("<If", len(pcm16), chunk.latency_ms or 0)
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
                    await websocket.send(json.dumps({"error": f"Unknown: {cmd}"}))
                    
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
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-s1", type=str, help="Speaker 1 voice")
    parser.add_argument("--prefix-s2", type=str, help="Speaker 2 voice")
    parser.add_argument("--port", type=int, default=3030)
    parser.add_argument("--seed", type=int, help="Random seed for reproducible generation")
    args = parser.parse_args()
    
    server = Dia2StreamingServer(
        model_repo="nari-labs/Dia2-2B",
        device="cuda",
        dtype="bfloat16",
        prefix_speaker_1=args.prefix_s1,
        prefix_speaker_2=args.prefix_s2,
        seed=args.seed,
    )
    
    await server.initialize()
    
    logger.info(f"WebSocket on ws://0.0.0.0:{args.port}")
    
    async with websockets.serve(
        lambda ws: handle_websocket(ws, server),
        "0.0.0.0",
        args.port,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
