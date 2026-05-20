import os
import sys
import io
import base64
import traceback
import numpy as np

import runpod

import torch
import torchaudio

print("[STARTUP] handler.py loaded — starting imports", flush=True)

try:
    from omnivoice.tts_api import OmniVoiceTTSEngine
    print("[STARTUP] OmniVoiceTTSEngine imported OK", flush=True)
except Exception as e:
    print("[STARTUP] IMPORT ERROR:\n" + traceback.format_exc(), file=sys.stderr, flush=True)
    raise

# -----------------------------
# One-time model load (worker warm state)
# -----------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "k2-fsa/OmniVoice")
DEVICE = os.environ.get("DEVICE", "cuda")

print(f"[STARTUP] MODEL_PATH={MODEL_PATH}, DEVICE={DEVICE}", flush=True)

try:
    engine = OmniVoiceTTSEngine()
    print("[STARTUP] OmniVoiceTTSEngine() created", flush=True)
    engine.tts_load(
        model_path=MODEL_PATH,
        reference_audio_path=None,
        reference_text=None,
        device=DEVICE,
        load_asr=False,
    )
    print("[STARTUP] tts_load() succeeded", flush=True)
except Exception as e:
    print("[STARTUP] MODEL LOAD ERROR:\n" + traceback.format_exc(), file=sys.stderr, flush=True)
    raise

# If you know sample_rate comes from the model, prefer that.
# Otherwise, set a default and/or read it from audio metadata.
DEFAULT_SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "24000"))


def _normalize_language(lang):
    if not lang:
        return None
    lang = str(lang).strip().lower()
    if lang in ("auto", "none", ""):
        return None
    return lang


def _encode_audio_bytes(waveform_np: np.ndarray, sample_rate: int, response_format: str) -> tuple[bytes, str]:
    """
    Returns (audio_bytes, content_type)
    waveform_np: float32 mono array [T] (or convertible)
    """
    response_format = (response_format or "mp3").lower().strip()

    # Ensure shape [1, T] float32 torch tensor
    if waveform_np.ndim != 1:
        waveform_np = np.squeeze(waveform_np)
    waveform_t = torch.from_numpy(waveform_np.astype(np.float32)).unsqueeze(0)

    buf = io.BytesIO()

    if response_format == "mp3":
        torchaudio.save(buf, waveform_t, sample_rate=sample_rate, format="mp3")
        return buf.getvalue(), "audio/mpeg"

    if response_format == "wav":
        torchaudio.save(buf, waveform_t, sample_rate=sample_rate, format="wav")
        return buf.getvalue(), "audio/wav"

    # NOTE: opus/aac may require ffmpeg-enabled backend depending on your environment.
    if response_format == "opus":
        torchaudio.save(buf, waveform_t, sample_rate=sample_rate, format="opus")
        return buf.getvalue(), "audio/opus"

    if response_format == "aac":
        torchaudio.save(buf, waveform_t, sample_rate=sample_rate, format="adts")
        return buf.getvalue(), "audio/aac"

    raise ValueError(f"Unsupported response_format: {response_format}")


def handler(event):
    """
    Expected event:
    {
      "input": {
        "input": "text to speak",               # required
        "voice": "instruct string",             # optional
        "language": "en" | "de" | "auto",       # optional
        "speed": 1.0,                           # optional
        "num_step": 16,                         # optional
        "response_format": "mp3"                # optional
      }
    }
    """
    payload = event.get("input", {}) or {}

    text = payload.get("input") or payload.get("text")
    if not text or not str(text).strip():
        return {"error": "Missing required field: input (text)"}

    instruct = payload.get("voice") or payload.get("instruct") or ""
    language = _normalize_language(payload.get("language"))
    speed = float(payload.get("speed", 1.0))
    num_step = int(payload.get("num_step", 24))
    response_format = payload.get("response_format", "mp3")

    # ---- Generate ----
    audio = engine._model.generate(
        text=str(text),
        language=language,
        instruct=str(instruct),
        speed=speed,
        num_step=num_step,
        position_temperature=0.0,
        class_temperature=0.0,
        guidance_scale=2.0,
        # add your other params here as needed
    )

    # Your note: audio[0] is waveform (numpy or torch)
    waveform = audio[0]

    if torch.is_tensor(waveform):
        waveform_np = waveform.detach().float().cpu().numpy()
    else:
        waveform_np = np.asarray(waveform, dtype=np.float32)

    sample_rate = int(payload.get("sample_rate", DEFAULT_SAMPLE_RATE))

    audio_bytes, content_type = _encode_audio_bytes(
        waveform_np=waveform_np,
        sample_rate=sample_rate,
        response_format=response_format,
    )

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "audio_base64": audio_b64,
        "content_type": content_type,
        "sample_rate": sample_rate,
        "response_format": response_format,
    }


runpod.serverless.start({"handler": handler})
