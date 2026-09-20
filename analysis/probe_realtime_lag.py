#!/usr/bin/env python3
"""Stream a recording at real-time pace and watch whether the server keeps up.

Lag is only visible when audio arrives no faster than a microphone produces
it. Replaying a file as fast as the socket accepts it measures throughput
instead and hides exactly the backlog being investigated, so this paces sends
against the wall clock the way the browser does.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_RATE = 16000
URL = "ws://127.0.0.1:8000/ws"


async def run(path, chunk_samples):
    audio = librosa.load(path, sr=SAMPLE_RATE)[0]
    period = chunk_samples / SAMPLE_RATE

    async with websockets.connect(URL) as ws:
        lags, start = [], time.monotonic()
        for i in range(0, len(audio) - chunk_samples, chunk_samples):
            target = start + (i / chunk_samples + 1) * period
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

            await ws.send(audio[i:i + chunk_samples].astype(np.float32).tobytes())
            reply = json.loads(await ws.recv())
            lags.append(reply.get("lag_seconds", float("nan")))

        lags = np.array(lags)
        audio_seconds = len(lags) * period
        print(f"{Path(path).name}: {audio_seconds:.1f}s streamed in "
              f"{chunk_samples / SAMPLE_RATE * 1000:.0f} ms chunks")
        print(f"  lag  first {lags[0] * 1000:6.0f} ms  "
              f"median {np.median(lags) * 1000:6.0f} ms  "
              f"final {lags[-1] * 1000:6.0f} ms  max {lags.max() * 1000:6.0f} ms")


asyncio.run(run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1600))
