"""
Streaming Audio Generation Plan for Dia2

This outlines how to modify the generation process to stream audio incrementally
rather than waiting for full generation to complete.

=============================================================================
KEY INSIGHTS
=============================================================================

1. MIMI CODEC CHARACTERISTICS:
   - Sample rate: 24,000 Hz
   - Frame rate: ~12.5 Hz (frames per second)
   - Samples per frame: ~1,920 samples (~80ms of audio per frame)
   - The codec uses multiple codebooks (typically 8) that are generated
     with different delays

2. DELAY PATTERN:
   - Each codebook has a delay (e.g., [0, 1, 2, 3, 4, 5, 6, 7])
   - This means codebook 0 is generated first, codebook 7 is delayed by 7 steps
   - We need to wait for all codebooks to "catch up" before decoding a frame
   - The max_delay determines the minimum latency before first audio output

3. GENERATION FLOW:
   - At each timestep t, we generate tokens for all codebooks at position t+1
   - Due to delays, the actual aligned frame at position t is only complete
     when we've generated up to t + max_delay

4. STREAMING OPPORTUNITY:
   - Once we have generated enough steps for the delays to "fill in",
     we can start decoding and streaming audio
   - We can decode in chunks (e.g., every N frames) to balance latency vs efficiency

=============================================================================
STREAMING ARCHITECTURE
=============================================================================

                    ┌─────────────────────────────────────────┐
                    │           Script/Text Input              │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │         State Machine + Tokenizer        │
                    │    (converts text to generation entries) │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     STREAMING GENERATION LOOP                        │
    │  ┌────────────────────────────────────────────────────────────────┐ │
    │  │  For each timestep t:                                          │ │
    │  │    1. Transformer forward → text logits, cb0 logits            │ │
    │  │    2. Sample text token, process state machine                 │ │
    │  │    3. Sample cb0 token                                         │ │
    │  │    4. Depformer stages → sample cb1..cbN tokens                │ │
    │  │    5. Store in audio_buf at position t+1                       │ │
    │  │                                                                │ │
    │  │    IF (t - max_delay) >= last_decoded_frame + chunk_size:      │ │
    │  │       → Extract completed frames                               │ │
    │  │       → Undelay and decode to PCM                              │ │
    │  │       → YIELD audio chunk                                      │ │
    │  └────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │         Audio Consumer / Player          │
                    │    (receives chunks as they're ready)    │
                    └─────────────────────────────────────────┘

=============================================================================
LATENCY ANALYSIS
=============================================================================

Minimum latency before first audio:
- max_delay frames (typically 7) × 80ms/frame = ~560ms
- Plus initial_padding frames (configurable, e.g., 6) × 80ms = ~480ms
- Plus chunk_size frames (configurable) × 80ms
- Total minimum: ~1-1.5 seconds with default settings

To reduce latency:
- Use smaller chunk_size (trade-off: more decode calls, less efficient)
- Reduce initial_padding (trade-off: may affect quality)
- The delay pattern is fixed by the model architecture

=============================================================================
IMPLEMENTATION PLAN
=============================================================================
"""

from __future__ import annotations
from typing import Iterator, Optional, Tuple, List
from dataclasses import dataclass
import torch


@dataclass
class StreamingConfig:
    """Configuration for streaming generation."""
    
    # Number of frames to accumulate before decoding and yielding
    # Smaller = lower latency, larger = more efficient
    chunk_frames: int = 8  # ~640ms of audio per chunk
    
    # Minimum frames to buffer before first output (must be >= max_delay)
    # This is automatically set to max(min_buffer_frames, max_delay)
    min_buffer_frames: int = 8
    
    # Whether to yield partial final chunk or wait for full chunk
    yield_partial_final: bool = True


@dataclass 
class AudioChunk:
    """A chunk of generated audio."""
    
    # PCM audio samples (float32, -1 to 1)
    samples: torch.Tensor
    
    # Sample rate (typically 24000)
    sample_rate: int
    
    # Frame indices this chunk covers (for debugging/timestamps)
    start_frame: int
    end_frame: int
    
    # Whether this is the final chunk
    is_final: bool = False
    
    # Word timestamps within this chunk: [(word, time_in_seconds), ...]
    timestamps: List[Tuple[str, float]] = None


def generate_streaming(
    self,  # Dia2 instance
    script: str,
    *,
    config: Optional['GenerationConfig'] = None,
    streaming_config: Optional[StreamingConfig] = None,
    verbose: bool = False,
    **overrides,
) -> Iterator[AudioChunk]:
    """
    Generate audio from script, yielding chunks as they become available.
    
    This is a generator function that yields AudioChunk objects incrementally.
    
    Usage:
        dia = Dia2.from_repo("nari-labs/Dia2-2B")
        for chunk in dia.generate_streaming("[S1] Hello world!"):
            # Process/play chunk.samples
            audio_player.write(chunk.samples.numpy())
    """
    
    streaming_config = streaming_config or StreamingConfig()
    
    # ========================================================================
    # PHASE 1: SETUP (same as regular generate)
    # ========================================================================
    
    runtime = self._ensure_runtime()
    # ... merge configs, parse script, build initial state ...
    # (Same as expanded_generate.py phases 1-3)
    
    # Key values we need:
    max_delay = max(runtime.audio_delays) if runtime.audio_delays else 0
    chunk_frames = streaming_config.chunk_frames
    min_buffer = max(streaming_config.min_buffer_frames, max_delay + 1)
    
    # ========================================================================
    # PHASE 2: STREAMING GENERATION LOOP
    # ========================================================================
    
    # Track decoding progress
    last_decoded_frame = -1  # Last frame we've decoded and yielded
    accumulated_timestamps = []  # Timestamps for current chunk
    
    with torch.inference_mode():
        for offset in range(max_context):
            t = start_step + offset
            
            # ... (same generation step as expanded_generate.py) ...
            # - Fill audio channels
            # - Transformer forward
            # - Sample text token, process state machine  
            # - Sample all codebook tokens
            # - Store in audio_buf
            
            # Track timestamps for this step
            if main_token == token_ids.new_word:
                # Record word and its frame position
                # (will be included in chunk timestamps)
                pass
            
            # ================================================================
            # STREAMING CHECK: Can we decode and yield a chunk?
            # ================================================================
            
            # The "safe" frame is the latest frame where all codebooks are filled
            # Due to delays, this is: current_step - max_delay
            safe_frame = t - max_delay
            
            # Calculate how many new frames we have since last decode
            frames_available = safe_frame - last_decoded_frame
            
            # Check if we have enough frames for a chunk
            should_yield = (
                frames_available >= chunk_frames and 
                safe_frame >= min_buffer
            )
            
            # Also yield if we've hit end of sequence
            is_final = (eos_cutoff is not None and t >= eos_cutoff - 1)
            if is_final and frames_available > 0 and streaming_config.yield_partial_final:
                should_yield = True
            
            if should_yield:
                # ============================================================
                # DECODE AND YIELD CHUNK
                # ============================================================
                
                # Determine frame range to decode
                decode_start = last_decoded_frame + 1
                decode_end = decode_start + chunk_frames
                if is_final:
                    decode_end = safe_frame + 1  # Include all remaining
                decode_end = min(decode_end, safe_frame + 1)
                
                # Extract the frames to decode
                # audio_buf shape: (branches, num_codebooks, total_steps)
                chunk_tokens = audio_buf[0, :, decode_start:decode_end]  # (codebooks, frames)
                
                # Undelay the frames
                # This aligns all codebooks by reversing the delay pattern
                aligned_chunk = undelay_chunk(chunk_tokens, runtime.audio_delays, token_ids.audio_pad)
                
                # Decode to PCM audio
                # Mimi expects shape: (batch, codebooks, frames)
                aligned_chunk = aligned_chunk.unsqueeze(0)
                pcm = runtime.mimi.decode(aligned_chunk)
                pcm = pcm[0, 0]  # Remove batch and channel dims
                
                # Build timestamps for this chunk
                chunk_timestamps = extract_timestamps_for_range(
                    accumulated_timestamps, 
                    decode_start, 
                    decode_end,
                    runtime.frame_rate
                )
                
                # Yield the chunk
                yield AudioChunk(
                    samples=pcm,
                    sample_rate=runtime.mimi.sample_rate,
                    start_frame=decode_start,
                    end_frame=decode_end,
                    is_final=is_final,
                    timestamps=chunk_timestamps,
                )
                
                # Update tracking
                last_decoded_frame = decode_end - 1
                
            # Check termination
            if is_final:
                break
    
    # ========================================================================
    # PHASE 3: FINAL FLUSH
    # ========================================================================
    
    # If there are remaining frames that weren't yielded, decode them now
    # (This handles the case where generation ended but we hadn't accumulated
    # enough frames for a full chunk)
    
    remaining_frames = safe_frame - last_decoded_frame
    if remaining_frames > 0:
        decode_start = last_decoded_frame + 1
        decode_end = safe_frame + 1
        
        chunk_tokens = audio_buf[0, :, decode_start:decode_end]
        aligned_chunk = undelay_chunk(chunk_tokens, runtime.audio_delays, token_ids.audio_pad)
        aligned_chunk = aligned_chunk.unsqueeze(0)
        pcm = runtime.mimi.decode(aligned_chunk)
        pcm = pcm[0, 0]
        
        yield AudioChunk(
            samples=pcm,
            sample_rate=runtime.mimi.sample_rate,
            start_frame=decode_start,
            end_frame=decode_end,
            is_final=True,
            timestamps=extract_timestamps_for_range(
                accumulated_timestamps, decode_start, decode_end, runtime.frame_rate
            ),
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def undelay_chunk(
    delayed_chunk: torch.Tensor,
    delays: List[int],
    pad_id: int,
) -> torch.Tensor:
    """
    Undelay a chunk of audio tokens.
    
    This is trickier than the full undelay because we're working with
    a sliding window. We need to handle boundary conditions.
    
    For streaming, we need to maintain state about "pending" tokens
    from previous chunks that haven't been aligned yet due to delays.
    
    Args:
        delayed_chunk: Shape (codebooks, frames) - tokens with delays applied
        delays: Per-codebook delay values
        pad_id: Padding token ID
    
    Returns:
        Shape (codebooks, output_frames) - aligned tokens
    """
    # For a chunk, the output length is reduced by max_delay
    # because we need tokens from "future" positions to align
    
    # IMPORTANT: For true streaming, we need to buffer tokens across chunks
    # This simplified version assumes we're decoding complete ranges
    
    codebooks, frames = delayed_chunk.shape
    max_delay = max(delays) if delays else 0
    
    # Output frames = input frames - max_delay (to ensure all codebooks have data)
    output_frames = max(0, frames - max_delay)
    
    if output_frames == 0:
        return delayed_chunk.new_full((codebooks, 0), pad_id)
    
    aligned = delayed_chunk.new_full((codebooks, output_frames), pad_id)
    
    for cb_idx, delay in enumerate(delays):
        # For codebook with delay D, position P in output comes from position P+D in input
        # But we've already applied delays during generation, so we reverse:
        # Output position P gets input position P + delay
        end_pos = delay + output_frames
        if end_pos <= frames:
            aligned[cb_idx] = delayed_chunk[cb_idx, delay:end_pos]
    
    return aligned


def extract_timestamps_for_range(
    all_timestamps: List[Tuple[str, int]],  # (word, frame_index)
    start_frame: int,
    end_frame: int,
    frame_rate: float,
) -> List[Tuple[str, float]]:
    """
    Extract timestamps that fall within the given frame range,
    converting frame indices to seconds relative to chunk start.
    """
    result = []
    for word, frame_idx in all_timestamps:
        if start_frame <= frame_idx < end_frame:
            # Convert to seconds relative to chunk start
            relative_frame = frame_idx - start_frame
            time_seconds = relative_frame / frame_rate
            result.append((word, time_seconds))
    return result


# ============================================================================
# ALTERNATIVE: ASYNC STREAMING
# ============================================================================

async def generate_streaming_async(
    self,
    script: str,
    *,
    config: Optional['GenerationConfig'] = None,
    streaming_config: Optional[StreamingConfig] = None,
    **overrides,
):
    """
    Async version that yields chunks and allows cooperative multitasking.
    
    This is useful for web servers or applications that need to handle
    multiple concurrent generations.
    
    Usage:
        async for chunk in dia.generate_streaming_async("[S1] Hello!"):
            await websocket.send(chunk.samples.numpy().tobytes())
    """
    import asyncio
    
    # Same setup as sync version...
    
    with torch.inference_mode():
        for offset in range(max_context):
            # ... generation step ...
            
            # Yield control periodically to allow other tasks to run
            if offset % 10 == 0:
                await asyncio.sleep(0)
            
            if should_yield:
                # ... decode chunk ...
                yield AudioChunk(...)


# ============================================================================
# CALLBACK-BASED STREAMING
# ============================================================================

def generate_with_callback(
    self,
    script: str,
    on_chunk: callable,  # Called with AudioChunk for each chunk
    on_complete: callable = None,  # Called when generation finishes
    *,
    config: Optional['GenerationConfig'] = None,
    streaming_config: Optional[StreamingConfig] = None,
    **overrides,
):
    """
    Generate audio and call a callback for each chunk.
    
    This is useful for integration with audio playback systems
    that use callbacks.
    
    Usage:
        def play_chunk(chunk):
            audio_device.write(chunk.samples.numpy())
        
        dia.generate_with_callback(
            "[S1] Hello world!",
            on_chunk=play_chunk,
        )
    """
    for chunk in generate_streaming(self, script, config=config, 
                                     streaming_config=streaming_config, **overrides):
        on_chunk(chunk)
    
    if on_complete:
        on_complete()


# ============================================================================
# WEBSOCKET STREAMING SERVER EXAMPLE
# ============================================================================

"""
Example FastAPI WebSocket server for streaming TTS:

from fastapi import FastAPI, WebSocket
import numpy as np

app = FastAPI()
dia = Dia2.from_repo("nari-labs/Dia2-2B", device="cuda")

@app.websocket("/tts")
async def tts_websocket(websocket: WebSocket):
    await websocket.accept()
    
    # Receive text to synthesize
    text = await websocket.receive_text()
    
    # Stream audio chunks back
    async for chunk in dia.generate_streaming_async(text):
        # Convert to bytes (16-bit PCM)
        pcm16 = (chunk.samples.cpu().numpy() * 32767).astype(np.int16)
        await websocket.send_bytes(pcm16.tobytes())
        
        # Optionally send timestamps as JSON
        if chunk.timestamps:
            await websocket.send_json({
                "type": "timestamps",
                "data": chunk.timestamps
            })
    
    await websocket.close()
"""


# ============================================================================
# IMPLEMENTATION CONSIDERATIONS
# ============================================================================

"""
1. CUDA GRAPH COMPATIBILITY:
   - CUDA graphs capture a fixed computation graph
   - Streaming decode calls may break graph capture
   - Solution: Either disable CUDA graphs for streaming, or capture
     separate graphs for generation and decoding

2. MEMORY MANAGEMENT:
   - The audio_buf grows with generation length
   - For very long generations, consider circular buffer or
     discarding already-decoded frames

3. MIMI DECODER EFFICIENCY:
   - Mimi may have startup overhead per decode call
   - Larger chunks = fewer decode calls = more efficient
   - But larger chunks = higher latency
   - Optimal chunk_frames depends on use case (8-16 frames is reasonable)

4. THREAD SAFETY:
   - The Dia2 model is not thread-safe
   - For concurrent requests, use separate model instances or
     implement request queuing

5. ERROR HANDLING:
   - Generation can fail mid-stream
   - Consumers should handle incomplete streams gracefully

6. BACKPRESSURE:
   - If consumer is slower than generation, chunks will queue up
   - Consider adding flow control or dropping frames for real-time use

7. FIRST-CHUNK LATENCY OPTIMIZATION:
   - The first chunk has unavoidable latency due to:
     a) Model warmup (first forward pass is slower)
     b) Delay pattern (must wait for all codebooks)
     c) initial_padding (quality vs latency trade-off)
   - For lowest latency: reduce initial_padding, use smaller first chunk

8. INCREMENTAL TEXT INPUT:
   - The README mentions "does not need entire text to produce audio"
   - This suggests the model can accept text incrementally
   - Future enhancement: allow appending to script during generation
   - This would enable true real-time conversation
"""


# ============================================================================
# FULL IMPLEMENTATION SKELETON
# ============================================================================

def generate_streaming_full(
    self,  # Dia2 instance  
    script: str,
    *,
    config: Optional['GenerationConfig'] = None,
    streaming_config: Optional[StreamingConfig] = None,
    prefix_speaker_1: Optional[str] = None,
    prefix_speaker_2: Optional[str] = None,
    include_prefix: Optional[bool] = None,
    verbose: bool = False,
    **overrides,
) -> Iterator[AudioChunk]:
    """
    Full implementation skeleton with all the pieces.
    """
    from .runtime.logger import RuntimeLogger
    from .generation import merge_generation_config, normalize_script
    from .runtime.script_parser import parse_script
    from .runtime.voice_clone import build_prefix_plan
    from .runtime.generator import build_initial_state, warmup_with_prefix
    from .runtime.guidance import apply_classifier_guidance, sample_audio_logits
    from .runtime.sampler import sample_token
    from .audio.grid import mask_audio_logits
    
    streaming_config = streaming_config or StreamingConfig()
    runtime = self._ensure_runtime()
    logger = RuntimeLogger(verbose)
    
    # Merge configuration
    merged_overrides = dict(overrides)
    if prefix_speaker_1 is not None:
        merged_overrides["prefix_speaker_1"] = prefix_speaker_1
    if prefix_speaker_2 is not None:
        merged_overrides["prefix_speaker_2"] = prefix_speaker_2
    if include_prefix is not None:
        merged_overrides["include_prefix"] = include_prefix
    merged = merge_generation_config(base=config or self.default_config, overrides=merged_overrides)
    
    # Parse script
    max_context = runtime.config.runtime.max_context_steps
    text = normalize_script(script)
    prefix_plan = build_prefix_plan(runtime, merged.prefix)
    
    entries = []
    if prefix_plan is not None:
        entries.extend(prefix_plan.entries)
    entries.extend(parse_script([text], runtime.tokenizer, runtime.constants, runtime.frame_rate))
    
    runtime.machine.initial_padding = merged.initial_padding
    
    # Build initial state
    state = runtime.machine.new_state(entries)
    gen_state = build_initial_state(runtime, prefix=prefix_plan)
    
    # Warmup with prefix if needed
    include_prefix_audio = bool(prefix_plan and merged.prefix and merged.prefix.include_audio)
    start_step = 0
    if prefix_plan is not None:
        start_step = warmup_with_prefix(runtime, prefix_plan, state, gen_state)
    
    # Setup for streaming
    step_tokens = gen_state.step_tokens
    audio_buf = gen_state.audio_buf
    branches = step_tokens.shape[0]
    token_ids = runtime.constants
    delay_tensor = runtime.audio_delay_tensor
    max_delay = int(delay_tensor.max().item()) if delay_tensor.numel() else 0
    
    chunk_frames = streaming_config.chunk_frames
    min_buffer = max(streaming_config.min_buffer_frames, max_delay + 1)
    
    cfg_active = merged.cfg_scale != 1.0
    flush_tail = max_delay + getattr(runtime.machine, "max_padding", 0)
    
    # Allocate buffers
    positions = torch.empty(1, 1, dtype=torch.long, device=runtime.device)
    main_tokens_buf = torch.empty(branches, dtype=torch.long, device=runtime.device)
    aux_tokens_buf = torch.empty(branches, dtype=torch.long, device=runtime.device)
    
    # Pre-allocate network buffers
    data_cfg = runtime.config.data
    logits_dtype = runtime.precision.logits
    text_logits_buf = torch.empty((branches, 1, data_cfg.action_vocab_size), 
                                   dtype=logits_dtype, device=runtime.device)
    cb0_logits_buf = torch.empty((branches, 1, data_cfg.audio_vocab_size),
                                  dtype=logits_dtype, device=runtime.device)
    dep_vocab = runtime.model.depformer.audio_vocab_limit or data_cfg.audio_vocab_size
    dep_logits_buf = [
        torch.empty((branches, 1, 1, dep_vocab), dtype=logits_dtype, device=runtime.device)
        for _ in range(runtime.model.depformer.num_depth)
    ]
    
    # Streaming state
    last_decoded_frame = start_step - 1
    first_word_frame = None
    eos_cutoff = None
    accumulated_timestamps = []
    
    positions_view = positions.expand(branches, -1)
    
    logger.event(f"Starting streaming generation (chunk_frames={chunk_frames}, min_buffer={min_buffer})")
    
    with torch.inference_mode():
        for offset in range(max_context):
            t = start_step + offset
            
            # Check termination
            if eos_cutoff is not None and t >= eos_cutoff:
                break
            if t + 1 >= audio_buf.shape[-1]:
                break
            
            # Reset depformer cache
            gen_state.decode.depformer.reset()
            positions.fill_(t)
            
            # Fill audio channels (with delays)
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
            
            # Transformer forward
            hidden_t, text_logits_t, cb0_logits_t, present = runtime.transformer_step(
                step_tokens, positions_view, gen_state.decode.transformer
            )
            text_logits_buf.copy_(text_logits_t)
            cb0_logits_buf.copy_(cb0_logits_t)
            gen_state.decode.transformer = present
            
            # Sample text token
            guided_text = apply_classifier_guidance(
                text_logits_buf, cfg_active, merged.cfg_scale, merged.cfg_filter_k
            )
            if guided_text.shape[0] > 1:
                guided_text = guided_text[:1]
            text_token = sample_token(
                guided_text, temp=merged.text.temperature, top_k=merged.text.top_k
            ).item()
            
            # Process state machine
            main_token, aux_token, _ = runtime.machine.process(t, state, text_token)
            second_token = aux_token if aux_token != -1 else token_ids.pad
            
            # Track first word and timestamps
            if first_word_frame is None and main_token == token_ids.new_word:
                first_word_frame = t - merged.initial_padding
            if main_token == token_ids.new_word and state.transcript:
                word, step = state.transcript[-1]
                accumulated_timestamps.append((word, step))
            
            step_tokens[:, 0, 0] = main_token
            step_tokens[:, 1, 0] = second_token
            
            # Sample cb0
            guided_cb0 = apply_classifier_guidance(
                cb0_logits_buf, cfg_active, merged.cfg_scale, merged.cfg_filter_k
            )
            if guided_cb0.shape[0] > 1:
                guided_cb0 = guided_cb0[:1]
            masked_cb0 = mask_audio_logits(guided_cb0, token_ids.audio_pad, token_ids.audio_bos)
            codebook_token = sample_audio_logits(masked_cb0, merged.audio.temperature, merged.audio.top_k)
            audio_buf[:, 0, t + 1] = codebook_token
            
            # Depformer stages
            prev_audio = codebook_token.expand(branches)
            main_tokens_buf.fill_(main_token)
            aux_tokens_buf.fill_(second_token)
            
            for stage in range(runtime.model.depformer.num_depth):
                logits_stage, dep_present = runtime.depformer_step(
                    prev_audio=prev_audio,
                    transformer_out=hidden_t,
                    stage_index=stage,
                    cache=gen_state.decode.depformer,
                    main_text=main_tokens_buf if stage == 0 else None,
                    second_text=aux_tokens_buf if stage == 0 else None,
                )
                dep_logits_buf[stage].copy_(logits_stage)
                gen_state.decode.depformer = dep_present
                
                dep_logits = apply_classifier_guidance(
                    dep_logits_buf[stage], cfg_active, merged.cfg_scale, merged.cfg_filter_k
                )
                if dep_logits.shape[0] > 1:
                    dep_logits = dep_logits[:1]
                stage_token = sample_audio_logits(
                    dep_logits, merged.audio.temperature, merged.audio.top_k
                )
                audio_buf[:, stage + 1, t + 1] = stage_token
                prev_audio = stage_token.expand(branches)
            
            # Check for EOS
            if eos_cutoff is None and state.end_step is not None:
                eos_cutoff = state.end_step + flush_tail
            
            # ================================================================
            # STREAMING: Check if we can yield a chunk
            # ================================================================
            
            safe_frame = t + 1 - max_delay  # Latest frame with all codebooks filled
            frames_available = safe_frame - last_decoded_frame
            is_final = (eos_cutoff is not None and t >= eos_cutoff - 1)
            
            should_yield = (
                frames_available >= chunk_frames and
                safe_frame >= min_buffer
            )
            
            if is_final and frames_available > 0 and streaming_config.yield_partial_final:
                should_yield = True
            
            if should_yield:
                decode_start = last_decoded_frame + 1
                decode_end = min(decode_start + chunk_frames, safe_frame + 1)
                if is_final:
                    decode_end = safe_frame + 1
                
                # Extract and undelay chunk
                chunk_tokens = audio_buf[0, :, decode_start:decode_end]
                aligned_chunk = undelay_chunk(
                    chunk_tokens, runtime.audio_delays, token_ids.audio_pad
                )
                
                if aligned_chunk.shape[-1] > 0:
                    # Decode to PCM
                    aligned_chunk = aligned_chunk.unsqueeze(0)
                    pcm = runtime.mimi.decode(aligned_chunk)
                    pcm = pcm[0, 0]
                    
                    # Extract timestamps for this chunk
                    chunk_timestamps = extract_timestamps_for_range(
                        accumulated_timestamps, decode_start, decode_end, runtime.frame_rate
                    )
                    
                    logger.event(f"Yielding chunk: frames {decode_start}-{decode_end}, "
                                f"{pcm.shape[-1]} samples")
                    
                    yield AudioChunk(
                        samples=pcm,
                        sample_rate=runtime.mimi.sample_rate,
                        start_frame=decode_start,
                        end_frame=decode_end,
                        is_final=is_final,
                        timestamps=chunk_timestamps,
                    )
                
                last_decoded_frame = decode_end - 1
            
            if is_final:
                break
    
    # Final flush
    safe_frame = (start_step + offset + 1) - max_delay if offset >= 0 else start_step
    remaining = safe_frame - last_decoded_frame
    
    if remaining > 0:
        decode_start = last_decoded_frame + 1
        decode_end = safe_frame + 1
        
        chunk_tokens = audio_buf[0, :, decode_start:decode_end]
        aligned_chunk = undelay_chunk(chunk_tokens, runtime.audio_delays, token_ids.audio_pad)
        
        if aligned_chunk.shape[-1] > 0:
            aligned_chunk = aligned_chunk.unsqueeze(0)
            pcm = runtime.mimi.decode(aligned_chunk)
            pcm = pcm[0, 0]
            
            chunk_timestamps = extract_timestamps_for_range(
                accumulated_timestamps, decode_start, decode_end, runtime.frame_rate
            )
            
            logger.event(f"Yielding final chunk: frames {decode_start}-{decode_end}")
            
            yield AudioChunk(
                samples=pcm,
                sample_rate=runtime.mimi.sample_rate,
                start_frame=decode_start,
                end_frame=decode_end,
                is_final=True,
                timestamps=chunk_timestamps,
            )
    
    logger.event("Streaming generation complete")
