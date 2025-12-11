"""
Test client for Dia2 Streaming TTS Server

Usage:
    python test_client.py
"""

import asyncio
import json
import struct
import wave
import time
import numpy as np

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    exit(1)


async def test_streaming():
    """Test the streaming TTS server."""
    
    uri = "wss://i9981txmurahrf-3030.proxy.runpod.net/"
    # uri = "ws://localhost:3030"
    
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as ws:
        # Test 1: Start a session
        print("\n=== Test 1: Starting session ===")
        
        start_time = time.time()
        await ws.send(json.dumps({
            "cmd": "start",
            "text": "[S1] Hello! This is a test of the Dia2 streaming text to speech system."
        }))
        
        all_samples = []
        sample_rate = 24000
        first_chunk_time = None
        first_chunk_latency_server = None
        
        while True:
            msg = await ws.recv()
            
            if isinstance(msg, bytes):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    client_latency = (first_chunk_time - start_time) * 1000
                
                # Parse header: 4 bytes sample count + 4 bytes latency
                num_samples = struct.unpack("<I", msg[:4])[0]
                server_latency = struct.unpack("<f", msg[4:8])[0]
                pcm16 = np.frombuffer(msg[8:], dtype=np.int16)
                samples = pcm16.astype(np.float32) / 32767.0
                all_samples.append(samples)
                
                if first_chunk_latency_server is None:
                    first_chunk_latency_server = server_latency
                    print(f"  *** First chunk: client={client_latency:.0f}ms, server={server_latency:.0f}ms ***")
                else:
                    print(f"  Received {len(samples)} samples ({len(samples)/sample_rate*1000:.0f}ms)")
            else:
                # JSON message
                data = json.loads(msg)
                print(f"  Event: {data}")
                if data.get("event") == "done":
                    break
        
        # Save first test
        if all_samples:
            audio = np.concatenate(all_samples)
            save_wav("test1_start.wav", audio, sample_rate)
            print(f"Saved test1_start.wav ({len(audio)/sample_rate:.2f}s)")
        
        # Test 2: Inject more text (should be faster!)
        print("\n=== Test 2: Injecting more text (should be low latency) ===")
        
        start_time = time.time()
        
        await ws.send(json.dumps({
            "cmd": "inject",
            "text": "[S1] This is the second sentence, injected after the first completed."
        }))
        
        all_samples = []
        first_chunk_time = None
        first_chunk_latency_server = None
        
        while True:
            msg = await ws.recv()
            
            if isinstance(msg, bytes):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    client_latency = (first_chunk_time - start_time) * 1000
                
                num_samples = struct.unpack("<I", msg[:4])[0]
                server_latency = struct.unpack("<f", msg[4:8])[0]
                pcm16 = np.frombuffer(msg[8:], dtype=np.int16)
                samples = pcm16.astype(np.float32) / 32767.0
                all_samples.append(samples)
                
                if first_chunk_latency_server is None:
                    first_chunk_latency_server = server_latency
                    print(f"  *** First chunk: client={client_latency:.0f}ms, server={server_latency:.0f}ms ***")
                else:
                    print(f"  Received {len(samples)} samples")
            else:
                data = json.loads(msg)
                print(f"  Event: {data}")
                if data.get("event") == "done":
                    break
        
        if all_samples:
            audio = np.concatenate(all_samples)
            save_wav("test2_inject.wav", audio, sample_rate)
            print(f"Saved test2_inject.wav ({len(audio)/sample_rate:.2f}s)")
        
        # Test 3: Another injection
        print("\n=== Test 3: Another injection ===")
        
        start_time = time.time()
        
        await ws.send(json.dumps({
            "cmd": "inject", 
            "text": "[S1] And here is a third sentence to test continuous generation."
        }))
        
        all_samples = []
        first_chunk_time = None
        first_chunk_latency_server = None
        
        while True:
            msg = await ws.recv()
            
            if isinstance(msg, bytes):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    client_latency = (first_chunk_time - start_time) * 1000
                
                num_samples = struct.unpack("<I", msg[:4])[0]
                server_latency = struct.unpack("<f", msg[4:8])[0]
                pcm16 = np.frombuffer(msg[8:], dtype=np.int16)
                samples = pcm16.astype(np.float32) / 32767.0
                all_samples.append(samples)
                
                if first_chunk_latency_server is None:
                    first_chunk_latency_server = server_latency
                    print(f"  *** First chunk: client={client_latency:.0f}ms, server={server_latency:.0f}ms ***")
            else:
                data = json.loads(msg)
                print(f"  Event: {data}")
                if data.get("event") == "done":
                    break
        
        if all_samples:
            audio = np.concatenate(all_samples)
            save_wav("test3_inject.wav", audio, sample_rate)
            print(f"Saved test3_inject.wav ({len(audio)/sample_rate:.2f}s)")
        
        print("\n=== Tests complete ===")


def save_wav(filename: str, audio: np.ndarray, sample_rate: int):
    """Save audio to WAV file."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm16.tobytes())


if __name__ == "__main__":
    asyncio.run(test_streaming())
