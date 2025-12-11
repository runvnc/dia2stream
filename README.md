# Dia2 Streaming TTS Server

Low-latency streaming TTS server using Dia2, optimized for voice call applications.

## Key Features

- **Persistent State**: Keeps model warm between turns for minimal latency
- **Text Injection**: Add new text without restarting generation
- **Frame-by-Frame Streaming**: 80ms audio chunks for real-time playback
- **Single Conversation Focus**: Optimized for one conversation per GPU

## Latency Characteristics

| Scenario | Latency |
|----------|--------|
| Cold start (first sentence) | ~1.44s (delay buffer fill) |
| Warm continuation (inject) | ~80-160ms |
| Per-frame streaming | 80ms chunks |

## Quick Start (RunPod / CUDA environment)

```bash
# Clone the repo
git clone https://github.com/runvnc/dia2stream.git
cd dia2stream

# Install PyTorch with CUDA first (if not already installed)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Then install other dependencies and run
uv run python server.py
```

Or if torch is already installed in your environment:

```bash
git clone https://github.com/runvnc/dia2stream.git
cd dia2stream
uv run --no-sync python server.py  # Skip dependency sync if torch already present
```

First run will download the Dia2-2B model (~4GB).

## Alternative: pip only

```bash
git clone https://github.com/runvnc/dia2stream.git
cd dia2stream
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers huggingface_hub websockets numpy soundfile
python server.py
```

## Architecture

```
                    WebSocket Client
                          |
                          v
                  +---------------+
                  |    Server     |
                  |  (server.py)  |
                  +---------------+
                          |
          +---------------+---------------+
          |                               |
          v                               v
   StreamingSession              Dia2 Model (GPU)
   - KV Cache (warm)             - Transformer
   - Audio Buffer                - Depformer  
   - State Machine               - Mimi Codec
```

## Usage

### Start the Server

```bash
python server.py
```

Server listens on `ws://0.0.0.0:8765`

### Test Client

```bash
python test_client.py
```

### WebSocket Protocol

**Commands (JSON):**

```json
// Start new session (cold start)
{"cmd": "start", "text": "[S1] Hello world!"}

// Inject more text (warm, low latency)
{"cmd": "inject", "text": "[S1] More text here."}

// Pause generation
{"cmd": "pause"}

// Reset session completely
{"cmd": "reset"}
```

**Responses:**

- Binary: `<4 bytes: sample count><PCM16 audio data>`
- JSON: `{"event": "done"}`, `{"event": "paused"}`, etc.

## Voice Call Integration

For optimal latency in a voice call:

1. **Call Start**: Initialize with greeting
   ```json
   {"cmd": "start", "text": "[S1] Hello, how can I help you today?"}
   ```

2. **User Speaks**: Pause TTS, run STT + LLM
   ```json
   {"cmd": "pause"}
   ```

3. **Agent Responds**: Inject LLM output (low latency!)
   ```json
   {"cmd": "inject", "text": "[S1] Sure, I can help with that."}
   ```

4. **Repeat** steps 2-3 for conversation turns

## Configuration

Edit `server.py` to adjust:

```python
self.default_config = GenerationConfig(
    cfg_scale=3.0,              # Classifier-free guidance
    text=SamplingConfig(temperature=0.7, top_k=50),
    audio=SamplingConfig(temperature=0.8, top_k=80),
    initial_padding=0,          # 0 for minimum latency
)

self.chunk_frames = 1  # 1 = decode every frame (80ms)
```

## File Structure

```
dia2stream/
├── dia2/              # Vendored Dia2 library
├── server.py          # Main streaming server
├── test_client.py     # Test client
├── pyproject.toml     # uv/pip configuration
└── README.md
```

## Requirements

- Python 3.10+
- CUDA GPU with ~8GB+ VRAM
- PyTorch with CUDA support

## Notes

- Model downloads on first run (~4GB)
- Single conversation per server instance
- For multiple concurrent conversations, run multiple server instances on different ports
