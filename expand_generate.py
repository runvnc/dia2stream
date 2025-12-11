\"\"\"
Expanded version of Dia2.generate() with all nested function logic inlined.

This file is for understanding the full generation flow in one place.
It is NOT meant to be executed - it contains pseudocode and inline expansions.
\"\"\"

from __future__ import annotations
from typing import Optional, Sequence, Tuple, List, Deque
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path

import torch
import numpy as np

# ============================================================================
# DATA STRUCTURES (from various modules)
# ============================================================================

@dataclass
class TokenIds:
    \"\"\"Token ID constants used throughout generation.\"\"\"
    card: int
    new_word: int
    pad: int
    bos: int
    zero: int
    spk1: int
    spk2: int
    audio_pad: int
    audio_bos: int
    ungenerated: int = -2


@dataclass
class Entry:
    \"\"\"A script entry representing a word/token to generate.\"\"\"
    tokens: List[int]
    text: str
    padding: int = 0


@dataclass
class State:
    \"\"\"State machine state tracking generation progress.\"\"\"
    entries: Deque[Entry]
    padding_budget: int
    forced_padding: int
    pending_tokens: Deque[int] = field(default_factory=deque)
    lookahead_tokens: Deque[int] = field(default_factory=deque)
    end_step: int | None = None
    consumption_times: List[int] = field(default_factory=list)
    transcript: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class GenerationState:
    \"\"\"Holds decode state, step tokens, and audio buffer during generation.\"\"\"
    decode: 'DecodeState'  # Contains transformer and depformer KV caches
    step_tokens: torch.Tensor  # Shape: (branches, channels, 1)
    audio_buf: torch.Tensor    # Shape: (branches, num_codebooks, total_steps)


@dataclass
class NetworkBuffers:
    \"\"\"Pre-allocated buffers for network outputs.\"\"\"
    text: torch.Tensor   # Text logits
    cb0: torch.Tensor    # First codebook logits
    dep: list[torch.Tensor]  # Depformer stage logits


# ============================================================================
# EXPANDED GENERATE FUNCTION
# ============================================================================

def expanded_generate(
    self,  # Dia2 instance
    script: str | Sequence[str],
    *,
    config: Optional['GenerationConfig'] = None,
    output_wav: Optional[str | Path] = None,
    prefix_speaker_1: Optional[str] = None,
    prefix_speaker_2: Optional[str] = None,
    include_prefix: Optional[bool] = None,
    verbose: bool = False,
    **overrides,
):
    \"\"\"
    EXPANDED generate() function with all nested logic inlined.
    
    This shows the complete flow from script input to audio output.
    \"\"\"
    
    # ========================================================================
    # PHASE 1: SETUP AND INITIALIZATION
    # ========================================================================
    
    # Ensure runtime is loaded (model, tokenizer, mimi codec)
    runtime = self._ensure_runtime()  # Returns RuntimeContext
    logger = RuntimeLogger(verbose)
    
    # Merge configuration with overrides
    merged_overrides = dict(overrides)
    if prefix_speaker_1 is not None:
        merged_overrides[\"prefix_speaker_1\"] = prefix_speaker_1
    if prefix_speaker_2 is not None:
        merged_overrides[\"prefix_speaker_2\"] = prefix_speaker_2
    if include_prefix is not None:
        merged_overrides[\"include_prefix\"] = include_prefix
    merged = merge_generation_config(base=config or self.default_config, overrides=merged_overrides)
    
    max_context = runtime.config.runtime.max_context_steps
    text = normalize_script(script)  # Convert to standard format
    
    # Build prefix plan for voice cloning (if applicable)
    prefix_plan = build_prefix_plan(runtime, merged.prefix)
    
    # Parse script into entries
    entries = []
    if prefix_plan is not None:
        entries.extend(prefix_plan.entries)
    entries.extend(parse_script([text], runtime.tokenizer, runtime.constants, runtime.frame_rate))
    
    runtime.machine.initial_padding = merged.initial_padding
    
    logger.event(
        f\"starting generation: max_context={max_context} cfg_scale={merged.cfg_scale:.2f} \"
        f\"device={self.device} dtype={self._dtype_pref}\"
    )
    
    # ========================================================================
    # PHASE 2: BUILD INITIAL STATE
    # ========================================================================
    # Inlined from: build_initial_state()
    
    # Create state machine state from entries
    state = runtime.machine.new_state(entries)
    # Expands to:
    #   state = State(
    #       entries=deque(entries),
    #       padding_budget=runtime.machine.initial_padding,
    #       forced_padding=runtime.machine.initial_padding,
    #   )
    
    cfg_active = merged.cfg_scale != 1.0
    if cfg_active:
        logger.event(f\"classifier-free guidance enabled (scale={merged.cfg_scale:.2f})\")
    else:
        logger.event(\"classifier-free guidance disabled (scale=1.0)\")
    
    # Build generation state with tensors
    dep_q = runtime.model.depformer.num_audio_channels  # Number of audio codebooks
    channels = 2 + dep_q  # 2 text channels + audio channels
    branches = 2  # Conditional and unconditional branches for CFG
    token_ids = runtime.constants
    
    # Initialize step tokens tensor
    step_tokens = torch.full(
        (branches, channels, 1),
        token_ids.pad,
        dtype=torch.long,
        device=runtime.device,
    )
    # Branch 0 (conditional): starts with BOS
    step_tokens[0, 0, 0] = token_ids.bos
    step_tokens[0, 1, 0] = token_ids.pad
    # Branch 1 (unconditional): starts with ZERO for CFG
    step_tokens[1, 0, 0] = token_ids.zero
    step_tokens[1, 1, 0] = token_ids.pad
    
    # Calculate total steps needed
    prefix_len = 0
    if prefix_plan is not None:
        # delay_frames() shifts audio tokens by per-codebook delays
        delayed = delay_frames(prefix_plan.aligned_tokens, runtime.audio_delays, token_ids.audio_pad)
        prefix_len = delayed.shape[1]
    limit = runtime.config.runtime.max_context_steps
    total_steps = max(limit + prefix_len + 1, limit)
    
    # Initialize model decode state (KV caches)
    decode_state = runtime.model.init_state(branches, runtime.device, total_steps)
    
    # Initialize audio buffer with \"ungenerated\" marker
    audio_buf = torch.full(
        (branches, dep_q, total_steps),
        token_ids.ungenerated,
        dtype=torch.long,
        device=runtime.device,
    )
    
    # Pre-fill audio buffer with prefix if present
    if prefix_plan is not None:
        delayed = delay_frames(prefix_plan.aligned_tokens, runtime.audio_delays, token_ids.audio_pad).to(runtime.device)
        audio_buf[0, :, : delayed.shape[1]] = delayed
        if branches > 1:
            audio_buf[1:, :, : delayed.shape[1]] = delayed
    
    gen_state = GenerationState(decode_state, step_tokens, audio_buf)
    
    # ========================================================================
    # PHASE 3: WARMUP WITH PREFIX (if applicable)
    # ========================================================================
    
    include_prefix_audio = bool(prefix_plan and merged.prefix and merged.prefix.include_audio)
    start_step = 0
    
    if prefix_plan is not None:
        logger.event(f\"warming up with prefix ({prefix_plan.aligned_frames} frames)\")
        
        # Inlined from: warmup_with_prefix()
        # This runs the model forward on prefix tokens without sampling
        tokens = prefix_plan.aligned_tokens.to(runtime.device)
        new_word_steps = set(prefix_plan.new_word_steps)
        positions = torch.empty(1, 1, dtype=torch.long, device=runtime.device)
        
        with torch.inference_mode():
            for t in range(prefix_plan.aligned_frames):
                positions.fill_(t)
                
                # Fill audio channels with delayed prefix tokens
                channels_count = tokens.shape[0]
                for cb in range(channels_count):
                    delay = runtime.audio_delays[cb] if cb < len(runtime.audio_delays) else 0
                    idx = t - delay
                    value = tokens[cb, idx] if idx >= 0 else runtime.constants.audio_bos
                    step_tokens[:, 2 + cb, 0] = value
                
                # Run transformer forward step (updates KV cache)
                hidden, text_logits, cb0_logits, present = runtime.model.transformer.forward_step(
                    step_tokens,
                    positions.expand(branches, -1),
                    gen_state.decode.transformer,
                )
                gen_state.decode.transformer = present
                
                # Force tokens based on prefix plan
                forced = runtime.constants.new_word if t in new_word_steps else runtime.constants.pad
                main_token, aux_token, _ = runtime.machine.process(t, state, forced, is_forced=True)
                second_token = runtime.constants.pad if aux_token == -1 else aux_token
                step_tokens[0, 0, 0] = main_token
                step_tokens[0, 1, 0] = second_token
                if branches > 1:
                    step_tokens[1:, 0, 0] = runtime.constants.zero
                    step_tokens[1:, 1, 0] = runtime.constants.pad
        
        start_step = max(prefix_plan.aligned_frames - 1, 0)
        
        if include_prefix_audio:
            logger.event(\"prefix audio will be kept in output\")
        else:
            logger.event(\"prefix audio trimmed from output\")
    
    # ========================================================================
    # PHASE 4: MAIN GENERATION LOOP
    # ========================================================================
    # Inlined from: run_generation_loop()
    
    step_tokens = gen_state.step_tokens
    audio_buf = gen_state.audio_buf
    branches = step_tokens.shape[0]
    max_context = runtime.config.runtime.max_context_steps
    
    # Pre-allocate tensors
    positions = torch.empty(1, 1, dtype=torch.long, device=runtime.device)
    main_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    aux_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    
    cfg_active = merged.cfg_scale != 1.0
    token_ids = runtime.constants
    delay_tensor = runtime.audio_delay_tensor
    max_delay = int(delay_tensor.max().item()) if delay_tensor.numel() else 0
    flush_tail = max_delay + getattr(runtime.machine, \"max_padding\", 0)
    
    first_word_frame: Optional[int] = None
    eos_cutoff: Optional[int] = None
    last_step = start_step - 1
    
    # Setup CUDA graph and torch.compile options
    use_graph = merged.use_cuda_graph and runtime.device.type == \"cuda\"
    use_torch_compile = merged.use_torch_compile and runtime.device.type == \"cuda\"
    
    if use_torch_compile:
        sample_token_fn = torch.compile(sample_token, dynamic=True, mode=\"max-autotune\", fullgraph=True)
        sample_audio_logits_fn = torch.compile(sample_audio_logits, dynamic=True, mode=\"max-autotune\", fullgraph=True)
    else:
        sample_token_fn = sample_token
        sample_audio_logits_fn = sample_audio_logits
    
    transformer_step = runtime.transformer_step
    depformer_step = runtime.depformer_step
    
    # Allocate network output buffers
    # Inlined from: _allocate_network_buffers()
    device = runtime.device
    logits_dtype = runtime.precision.logits
    data_cfg = runtime.config.data
    text_logits_buf = torch.empty((branches, 1, data_cfg.action_vocab_size), dtype=logits_dtype, device=device)
    cb0_logits_buf = torch.empty((branches, 1, data_cfg.audio_vocab_size), dtype=logits_dtype, device=device)
    dep_vocab = runtime.model.depformer.audio_vocab_limit or data_cfg.audio_vocab_size
    dep_logits_buf = [
        torch.empty((branches, 1, 1, dep_vocab), dtype=logits_dtype, device=device)
        for _ in range(runtime.model.depformer.num_depth)
    ]
    buffers = NetworkBuffers(text=text_logits_buf, cb0=cb0_logits_buf, dep=dep_logits_buf)
    
    positions_view = positions.expand(branches, -1)
    transformer_capture = None
    dep_captures = None
    
    if use_graph:
        # Ensure cuBLAS is ready for CUDA graphs
        tmp = torch.empty((1, 1), device=device, dtype=torch.float32)
        torch.matmul(tmp, tmp)
        torch.cuda.synchronize()
    
    processed_steps = 0
    report_interval = 12
    
    # ========================================================================
    # THE MAIN AUTOREGRESSIVE LOOP
    # ========================================================================
    
    with torch.inference_mode():
        for offset in range(max_context):
            if use_torch_compile:
                torch.compiler.cudagraph_mark_step_begin()
            
            t = start_step + offset  # Current timestep
            
            # Check termination conditions
            if eos_cutoff is not None and t >= eos_cutoff:
                break
            if t + 1 >= audio_buf.shape[-1]:
                break
            
            # Reset depformer KV cache for this step
            gen_state.decode.depformer.reset()
            
            positions.fill_(t)
            
            # ----------------------------------------------------------------
            # FILL AUDIO CHANNELS
            # ----------------------------------------------------------------
            # Inlined from: _fill_audio_channels()
            # Each codebook has a different delay, so we look back appropriately
            num_audio_channels = delay_tensor.numel()
            if num_audio_channels > 0:
                target = step_tokens[:, 2 : 2 + num_audio_channels, 0]
                if t < audio_buf.shape[-1]:
                    target.copy_(audio_buf[:, :num_audio_channels, t])
                else:
                    target.fill_(token_ids.audio_bos)
                # Mask out channels where delay > current step
                mask = delay_tensor > t
                mask_expanded = mask.unsqueeze(0).expand_as(target)
                target.copy_(torch.where(mask_expanded, token_ids.audio_bos, target))
            
            # Set unconditional branch tokens for CFG
            if branches > 1:
                step_tokens[1:, 0, 0] = token_ids.zero
                step_tokens[1:, 1, 0] = token_ids.pad
            
            # ----------------------------------------------------------------
            # TRANSFORMER FORWARD STEP
            # ----------------------------------------------------------------
            # Runs the main transformer to get text and first codebook logits
            
            # (Simplified - actual code has CUDA graph capture logic)
            hidden_t, text_logits_t, cb0_logits_t, present = transformer_step(
                step_tokens,
                positions_view,
                gen_state.decode.transformer,
            )
            buffers.text.copy_(text_logits_t)
            buffers.cb0.copy_(cb0_logits_t)
            gen_state.decode.transformer = present
            
            # ----------------------------------------------------------------
            # SAMPLE TEXT TOKEN
            # ----------------------------------------------------------------
            # Inlined from: apply_classifier_guidance() and sample_token()
            
            # Apply classifier-free guidance to text logits
            if cfg_active:
                conditional = buffers.text[0:1]
                unconditional = buffers.text[1:2]
                cond32 = conditional.to(torch.float32)
                uncond32 = unconditional.to(torch.float32)
                # Interpolate: guided = uncond + scale * (cond - uncond)
                guided_text = torch.lerp(uncond32, cond32, merged.cfg_scale)
                # Optional top-k filtering for CFG
                if merged.cfg_filter_k > 0 and guided_text.shape[-1] > 0:
                    k = min(merged.cfg_filter_k, guided_text.shape[-1])
                    threshold = torch.topk(guided_text, k=k, dim=-1, sorted=False).values[..., -1:]
                    mask = guided_text >= threshold
                    neg_inf = torch.full_like(cond32, float(\"-inf\"))
                    cond32 = torch.where(mask, cond32, neg_inf)
                guided_text = cond32.to(conditional.dtype)
            else:
                guided_text = buffers.text
            
            if guided_text.shape[0] > 1:
                guided_text = guided_text[:1]
            
            # Sample text token
            # Inlined from: sample_token()
            logits32 = guided_text.to(torch.float32)
            if merged.text.temperature <= 0.0:
                text_token = torch.argmax(logits32, dim=-1, keepdim=True).item()
            else:
                probs = torch.softmax(logits32 / max(merged.text.temperature, 1e-6), dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                probs = torch.clamp_min(probs, 0.0)
                flat = probs.reshape(-1, probs.shape[-1])
                norm = flat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                flat = flat / norm
                vocab = flat.shape[-1]
                if merged.text.top_k > 0 and merged.text.top_k < vocab:
                    topv, indices = torch.topk(flat, merged.text.top_k, dim=-1)
                    topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    draws = torch.multinomial(topv, num_samples=1)
                    text_token = torch.gather(indices, dim=-1, index=draws).item()
                else:
                    text_token = torch.multinomial(flat, num_samples=1).item()
            
            # ----------------------------------------------------------------
            # STATE MACHINE PROCESSING
            # ----------------------------------------------------------------
            # Inlined from: runtime.machine.process()
            # This determines what token to actually output based on script state
            
            # Sanitize token (map 0->pad, 1->new_word)
            if text_token == 1:
                text_token = token_ids.new_word
            elif text_token == 0:
                text_token = token_ids.pad
            if text_token not in (token_ids.new_word, token_ids.pad):
                text_token = token_ids.pad
            
            # Enforce constraints (pending tokens, forced padding, etc.)
            if state.pending_tokens:
                text_token = token_ids.pad
            elif state.forced_padding > 0:
                text_token = token_ids.pad
            elif state.padding_budget <= 0 and text_token != token_ids.new_word:
                text_token = token_ids.new_word
            
            # Handle new_word token - consume next entry from script
            consumed_new_word = False
            if text_token == token_ids.new_word:
                if state.entries:
                    entry = state.entries.popleft()
                    state.consumption_times.append(t)
                    if entry.tokens:
                        state.transcript.append((entry.text, t))
                        state.pending_tokens.extend(entry.tokens)
                        state.padding_budget = runtime.machine.max_padding
                    else:
                        text_token = token_ids.pad
                    state.forced_padding = entry.padding
                    consumed_new_word = True
                else:
                    text_token = token_ids.pad
                    if state.end_step is None:
                        state.end_step = t
            
            # Select output token
            if text_token == token_ids.pad:
                if state.padding_budget > 0:
                    state.padding_budget -= 1
                if state.forced_padding > 0:
                    state.forced_padding -= 1
                if state.pending_tokens:
                    main_token = state.pending_tokens.popleft()
                else:
                    main_token = token_ids.pad
            elif text_token == token_ids.new_word:
                main_token = token_ids.new_word
            else:
                main_token = text_token
            
            # Handle second stream (for dual-speaker scenarios)
            aux_token = -1  # Simplified - actual logic handles second stream
            second_token = aux_token if aux_token != -1 else token_ids.pad
            
            # Track first word frame for timestamp alignment
            if first_word_frame is None and main_token == token_ids.new_word:
                first_word_frame = t - merged.initial_padding
            
            # Update step tokens with processed values
            step_tokens[:, 0, 0] = main_token
            step_tokens[:, 1, 0] = second_token
            
            # ----------------------------------------------------------------
            # SAMPLE FIRST CODEBOOK (CB0)
            # ----------------------------------------------------------------
            
            # Apply CFG to cb0 logits (same pattern as text)
            if cfg_active:
                conditional = buffers.cb0[0:1]
                unconditional = buffers.cb0[1:2]
                guided_cb0 = torch.lerp(
                    unconditional.to(torch.float32),
                    conditional.to(torch.float32),
                    merged.cfg_scale
                ).to(conditional.dtype)
            else:
                guided_cb0 = buffers.cb0
            
            if guided_cb0.shape[0] > 1:
                guided_cb0 = guided_cb0[:1]
            
            # Mask out pad and bos tokens from audio logits
            # Inlined from: mask_audio_logits()
            if guided_cb0.shape[-1] > 0:
                max_idx = guided_cb0.shape[-1] - 1
                targets = [idx for idx in (token_ids.audio_pad, token_ids.audio_bos) if 0 <= idx <= max_idx]
                if targets:
                    masked_cb0 = guided_cb0.clone()
                    neg_inf = torch.finfo(masked_cb0.dtype).min
                    for idx in targets:
                        masked_cb0[..., idx] = neg_inf
                else:
                    masked_cb0 = guided_cb0
            else:
                masked_cb0 = guided_cb0
            
            # Sample first codebook token
            codebook_token = sample_audio_logits_fn(masked_cb0, merged.audio.temperature, merged.audio.top_k)
            audio_buf[:, 0, t + 1] = codebook_token
            
            # ----------------------------------------------------------------
            # DEPFORMER: SAMPLE REMAINING CODEBOOKS
            # ----------------------------------------------------------------
            # The depformer autoregressively generates codebooks 1..N
            # conditioned on the transformer hidden state and previous codebooks
            
            prev_audio = codebook_token.expand(branches)
            main_tokens.fill_(main_token)
            aux_tokens.fill_(second_token)
            
            for stage in range(runtime.model.depformer.num_depth):
                # Run depformer stage
                # Inlined from: _execute_depformer_stage()
                logits_stage, dep_present = depformer_step(
                    prev_audio=prev_audio,
                    transformer_out=hidden_t,
                    stage_index=stage,
                    cache=gen_state.decode.depformer,
                    main_text=main_tokens if stage == 0 else None,
                    second_text=aux_tokens if stage == 0 else None,
                )
                buffers.dep[stage].copy_(logits_stage)
                gen_state.decode.depformer = dep_present
                
                # Apply CFG to depformer logits
                dep_logits = buffers.dep[stage]
                if cfg_active:
                    conditional = dep_logits[0:1]
                    unconditional = dep_logits[1:2]
                    dep_logits = torch.lerp(
                        unconditional.to(torch.float32),
                        conditional.to(torch.float32),
                        merged.cfg_scale
                    ).to(conditional.dtype)
                
                if dep_logits.shape[0] > 1:
                    dep_logits = dep_logits[:1]
                
                # Sample this codebook
                stage_token = sample_audio_logits_fn(
                    dep_logits,
                    merged.audio.temperature,
                    merged.audio.top_k,
                )
                audio_buf[:, stage + 1, t + 1] = stage_token
                prev_audio = stage_token.expand(branches)
            
            # ----------------------------------------------------------------
            # END OF STEP BOOKKEEPING
            # ----------------------------------------------------------------
            
            last_step = t
            
            # Check if we've reached end of script
            if eos_cutoff is None and state.end_step is not None:
                eos_cutoff = state.end_step + flush_tail
            
            processed_steps = offset + 1
            if logger and processed_steps % report_interval == 0:
                logger.progress(processed_steps, max_context)
    
    # End of generation loop
    
    if logger and processed_steps and processed_steps % report_interval != 0:
        logger.progress(processed_steps, max_context)
    
    if first_word_frame is None:
        first_word_frame = start_step
    
    # Trim audio buffer to actual generated length
    if last_step < start_step:
        limit = min(start_step + 1, audio_buf.shape[-1])
    else:
        limit = min(last_step + 2, audio_buf.shape[-1])
    
    # Replace ungenerated markers with pad tokens
    trimmed = audio_buf[:, :, :limit]
    pad = torch.full_like(trimmed, token_ids.audio_pad)
    trimmed = torch.where(trimmed == token_ids.ungenerated, pad, trimmed)
    
    # ========================================================================
    # PHASE 5: UNDELAY FRAMES
    # ========================================================================
    # Inlined from: undelay_frames()
    # Reverses the per-codebook delays applied during generation
    
    delayed = trimmed[0]  # Take first branch (conditional)
    delays = runtime.audio_delays
    pad_id = runtime.constants.audio_pad
    
    channels, total = delayed.shape
    max_delay = max(delays) if delays else 0
    target_len = max(0, total - max_delay)
    aligned = delayed.new_full((channels, target_len), pad_id)
    
    for idx, delay in enumerate(delays):
        # Each codebook was delayed by 'delay' steps during generation
        # Now we shift it back to align all codebooks
        aligned[idx] = delayed[idx, delay : delay + target_len]
    
    aligned = aligned.unsqueeze(0)  # Add batch dimension back
    
    # ========================================================================
    # PHASE 6: CROP PREFIX AUDIO (if not including)
    # ========================================================================
    
    crop = 0 if include_prefix_audio else max(first_word_frame, 0)
    if crop > 0 and crop < aligned.shape[-1]:
        aligned = aligned[:, :, crop:]
    elif crop >= aligned.shape[-1]:
        crop = 0
    
    logger.event(f\"decoding {aligned.shape[-1]} Mimi frames\")
    
    # ========================================================================
    # PHASE 7: DECODE AUDIO
    # ========================================================================
    # Inlined from: decode_audio()
    # Converts discrete audio tokens back to waveform using Mimi codec
    
    if aligned.shape[-1] == 0:
        waveform = torch.zeros(0, device=runtime.device)
    else:
        with torch.inference_mode():
            # Mimi decoder: tokens -> PCM waveform
            pcm = runtime.mimi.decode(aligned.to(runtime.device))
            waveform = pcm[0, 0]  # Remove batch and channel dims
    
    # ========================================================================
    # PHASE 8: SAVE OUTPUT AND BUILD RESULT
    # ========================================================================
    
    if output_wav is not None:
        # Inlined from: write_wav()
        path = Path(output_wav)
        path.parent.mkdir(parents=True, exist_ok=True)
        audio_np = waveform.detach().cpu().numpy()
        audio_np = np.clip(audio_np, -1.0, 1.0)
        pcm16 = (audio_np * 32767.0).astype(np.int16)
        import wave
        with wave.open(str(path), \"wb\") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(runtime.mimi.sample_rate)
            handle.writeframes(pcm16.tobytes())
        
        duration = waveform.shape[-1] / max(runtime.mimi.sample_rate, 1)
        logger.event(f\"saved {output_wav} ({duration:.2f}s)\")
    
    # Build timestamps from transcript
    frame_rate = max(runtime.frame_rate, 1.0)
    prefix_entry_count = len(prefix_plan.entries) if prefix_plan is not None else 0
    transcript_entries = state.transcript
    
    if prefix_plan is not None and not include_prefix_audio:
        if len(transcript_entries) > prefix_entry_count:
            transcript_entries = transcript_entries[prefix_entry_count:]
        else:
            transcript_entries = []
    
    timestamps = []
    for word, step in transcript_entries:
        adj = step - crop
        if adj < 0:
            continue
        timestamps.append((word, adj / frame_rate))
    
    logger.event(f\"generation finished in {logger.elapsed():.2f}s\")
    
    return GenerationResult(aligned, waveform, runtime.mimi.sample_rate, timestamps)


# ============================================================================
# HELPER FUNCTIONS (referenced but not fully inlined above)
# ============================================================================

def delay_frames(aligned: torch.Tensor, delays: Sequence[int], pad_id: int) -> torch.Tensor:
    \"\"\"
    Apply per-codebook delays to aligned audio tokens.
    
    Each codebook is shifted forward by its delay amount, with padding added.
    This is used because different codebooks represent different time scales
    in the neural audio codec.
    
    Args:
        aligned: Shape (channels, total) - aligned audio tokens
        delays: Per-channel delay amounts
        pad_id: Token ID to use for padding
    
    Returns:
        Shape (channels, total + max_delay) - delayed tokens
    \"\"\"
    channels, total = aligned.shape
    max_delay = max(delays) if delays else 0
    out = aligned.new_full((channels, total + max_delay), pad_id)
    for idx, delay in enumerate(delays):
        out[idx, delay : delay + total] = aligned[idx]
    return out


def sample_token(logits: torch.Tensor, temp: float, top_k: int = 0) -> torch.Tensor:
    \"\"\"
    Sample a token from logits using temperature and optional top-k.
    
    Args:
        logits: Raw logits from model
        temp: Temperature (0 = greedy, higher = more random)
        top_k: If > 0, only sample from top k tokens
    
    Returns:
        Sampled token index
    \"\"\"
    logits32 = logits.to(torch.float32)
    if temp <= 0.0:
        return torch.argmax(logits32, dim=-1, keepdim=True)
    
    probs = torch.softmax(logits32 / max(temp, 1e-6), dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.clamp_min(probs, 0.0)
    
    flat = probs.reshape(-1, probs.shape[-1])
    norm = flat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    flat = flat / norm
    
    # Handle zero probability case
    zero_mask = norm <= 0
    filler = torch.zeros_like(flat)
    filler[..., 0] = 1.0
    mask = zero_mask.expand_as(flat)
    flat = torch.where(mask, filler, flat)
    
    vocab = flat.shape[-1]
    if top_k > 0 and top_k < vocab:
        topv, indices = torch.topk(flat, top_k, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        draws = torch.multinomial(topv, num_samples=1)
        picks = torch.gather(indices, dim=-1, index=draws)
    else:
        picks = torch.multinomial(flat, num_samples=1)
    
    return picks.reshape(probs.shape[:-1] + (1,))


def sample_audio_logits(logits: torch.Tensor, temp: float, top_k: int) -> torch.Tensor:
    \"\"\"Sample a single audio token from logits.\"\"\"
    return sample_token(logits, temp=temp, top_k=top_k).view(1)


# ============================================================================
# SUMMARY OF GENERATION FLOW
# ============================================================================
\"\"\"
1. SETUP
   - Load runtime (model, tokenizer, Mimi codec)
   - Parse script into entries (words/tokens to generate)
   - Merge configuration options

2. INITIALIZE STATE
   - Create state machine state from script entries
   - Allocate step_tokens tensor (branches x channels x 1)
   - Allocate audio_buf tensor (branches x codebooks x max_steps)
   - Initialize KV caches for transformer and depformer

3. WARMUP (if prefix/voice cloning)
   - Run model forward on prefix tokens without sampling
   - Updates KV caches to condition on prefix

4. MAIN GENERATION LOOP (for each timestep t):
   a. Fill audio channels with delayed tokens from previous steps
   b. Run transformer forward step -> get hidden state, text logits, cb0 logits
   c. Apply classifier-free guidance to text logits
   d. Sample text token (new_word or pad)
   e. Process through state machine (consume script entries, manage padding)
   f. Apply CFG to cb0 logits, mask special tokens, sample first codebook
   g. For each depformer stage:
      - Run depformer conditioned on hidden state and previous codebooks
      - Apply CFG, sample next codebook token
   h. Store all codebook tokens in audio_buf at position t+1
   i. Check for end-of-sequence

5. POST-PROCESS
   - Trim audio buffer to actual length
   - Undelay frames (reverse per-codebook delays)
   - Optionally crop prefix audio

6. DECODE
   - Pass aligned tokens through Mimi decoder -> PCM waveform

7. OUTPUT
   - Optionally save to WAV file
   - Build timestamps from transcript
   - Return GenerationResult with tokens, waveform, sample rate, timestamps
\"\"\"
