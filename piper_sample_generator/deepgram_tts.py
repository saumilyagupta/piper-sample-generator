#!/usr/bin/env python3
"""Generate wake word samples using Deepgram's Aura text-to-speech voices.

Piper's LibriTTS generator gives many speakers but a single synthesis model,
so its samples share one set of acoustic quirks. Mixing in Deepgram's Aura
voices adds genuinely different vocoders, accents and speaking styles, which
is what the detector needs in order to generalise past synthetic audio.
"""

import logging
import os
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Union
from urllib import error, request

import numpy as np

_LOGGER = logging.getLogger(__name__)

MODELS_URL = "https://api.deepgram.com/v1/models"
SPEAK_URL = "https://api.deepgram.com/v1/speak"
SAMPLE_RATE = 16000


class DeepgramError(RuntimeError):
    """Raised when the Deepgram API cannot fulfil a request."""


def get_api_key(explicit: Optional[str] = None) -> str:
    """Resolve the API key from an argument or DEEPGRAM_API_KEY."""
    key = explicit or os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise DeepgramError(
            "No Deepgram API key. Pass --api-key or set DEEPGRAM_API_KEY."
        )
    return key


def _get_json(url: str, api_key: str) -> dict:
    import json

    req = request.Request(url, headers={"Authorization": f"Token {api_key}"})
    with request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def list_english_voices(api_key: str) -> List[str]:
    """Return every English Aura voice the account can use."""
    data = _get_json(MODELS_URL, api_key)

    voices = []
    for model in data.get("tts", []):
        languages = [str(lang) for lang in model.get("languages", [])]
        if any(lang.startswith("en") for lang in languages):
            name = model.get("canonical_name") or model.get("name")
            if name:
                voices.append(name)

    return sorted(set(voices))


def synthesize(text: str, voice: str, api_key: str, retries: int = 3) -> np.ndarray:
    """Synthesize one phrase, returning int16 PCM at 16kHz.

    Requests raw PCM rather than a WAV container: Deepgram streams the
    container with a placeholder length in its header, so the resulting file
    reports a nonsense duration.
    """
    import json

    url = (
        f"{SPEAK_URL}?model={voice}&encoding=linear16"
        f"&sample_rate={SAMPLE_RATE}&container=none"
    )
    body = json.dumps({"text": text}).encode("utf-8")

    last_error = None
    for attempt in range(retries):
        try:
            req = request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Token {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with request.urlopen(req, timeout=60) as resp:
                return np.frombuffer(resp.read(), dtype=np.int16)
        except error.HTTPError as e:
            last_error = e
            # Back off on rate limits and transient server errors.
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise DeepgramError(f"{voice}: HTTP {e.code} {e.reason}") from e
        except error.URLError as e:
            last_error = e
            time.sleep(2 ** attempt)

    raise DeepgramError(f"{voice}: failed after {retries} attempts ({last_error})")


def _write_wav(path: Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(pcm.tobytes())


def generate_samples(
    phrases: Iterable[str],
    output_dir: Union[str, Path],
    api_key: Optional[str] = None,
    voices: Optional[List[str]] = None,
    max_workers: int = 8,
    prefix: str = "dg",
) -> int:
    """Synthesize every phrase with every voice into output_dir."""
    api_key = get_api_key(api_key)
    phrases = [p for p in phrases if p and p.strip()]

    if voices is None:
        voices = list_english_voices(api_key)
        _LOGGER.info(f"Found {len(voices)} English Deepgram voices")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(p, v) for v in voices for p in phrases]
    _LOGGER.info(f"Synthesizing {len(jobs)} clips ({len(phrases)} phrases x {len(voices)} voices)")

    written = 0
    failed = 0

    def run(index_job):
        index, (text, voice) = index_job
        pcm = synthesize(text, voice, api_key)
        _write_wav(output_dir / f"{prefix}_{index}.wav", pcm)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run, item): item for item in enumerate(jobs)}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
                written += 1
            except Exception as e:
                failed += 1
                if failed <= 5:
                    _LOGGER.warning(f"Synthesis failed: {e}")

            if done % 250 == 0:
                _LOGGER.info(f"  {done}/{len(jobs)} done ({failed} failed)")

    _LOGGER.info(f"Wrote {written} clips to {output_dir} ({failed} failed)")
    return written
