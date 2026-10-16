"""On-demand local Santali speech generation for Bhasha Setu.

The translation adapter and speech generator are separate models. This module
loads the optional AI4Bharat Parler-TTS model only when a reviewer requests a
Santali audio draft, then reuses it for later lessons.
"""
from __future__ import annotations

import os
import re
import json
import sys
import threading
import time
from contextlib import redirect_stdout
from io import BytesIO
from pathlib import Path

MODEL_ID = os.environ.get("BS_SANTALI_TTS_MODEL", "ai4bharat/indic-parler-tts")
MODEL_CAPTION = (
    "A fluent Santali speaker reads the Santali lesson slowly and clearly "
    "for primary school students. The voice is steady and the audio is very clear."
)

_model = None
_tokenizer = None
_description_tokenizer = None
_description_inputs = None
_torch = None
_device = None
_last_generation_metrics = {}
_model_lock = threading.Lock()
_generation_lock = threading.Lock()


def _load_model():
    global _model, _tokenizer, _description_tokenizer, _description_inputs, _torch, _device
    if _model is not None:
        return _model, _tokenizer, _description_tokenizer, _torch, _device
    with _model_lock:
        if _model is not None:
            return _model, _tokenizer, _description_tokenizer, _torch, _device
        try:
            print("[Santali TTS] importing CPU/GPU runtime", file=sys.stderr, flush=True)
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                # Large CPU TTS models can slow down when PyTorch creates more
                # worker threads than the laptop can use efficiently. Allow a
                # local override for machines that benchmark better at another
                # setting; default to at most eight logical CPU threads.
                try:
                    cpu_threads = int(os.environ.get("BS_SANTALI_TTS_CPU_THREADS", "0"))
                except ValueError as exc:
                    raise RuntimeError("BS_SANTALI_TTS_CPU_THREADS must be a positive integer.") from exc
                if not cpu_threads:
                    cpu_threads = min(os.cpu_count() or 4, 8)
                if cpu_threads < 1:
                    raise RuntimeError("BS_SANTALI_TTS_CPU_THREADS must be a positive integer.")
                torch.set_num_threads(cpu_threads)
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    # PyTorch only allows this before its first parallel work;
                    # the default inter-op setting is safe if it is already set.
                    pass
                print(f"[Santali TTS] CPU inference threads: {cpu_threads}", file=sys.stderr, flush=True)
            # On Windows, safetensors' default mmap reader can hit a native
            # PyTorch access violation while loading large checkpoints. Use its
            # pread backend in this isolated process to avoid mmap pointer handoff.
            if os.name == "nt":
                try:
                    import safetensors

                    original_safe_open = safetensors.safe_open

                    def safe_open_with_pread(*args, **kwargs):
                        kwargs["backend"] = "pread"
                        return original_safe_open(*args, **kwargs)

                    safetensors.safe_open = safe_open_with_pread
                    print("[Santali TTS] using safetensors pread backend on Windows", file=sys.stderr, flush=True)
                    for module_name in (
                        "safetensors.torch",
                        "transformers.modeling_utils",
                        "accelerate.utils.modeling",
                        "accelerate.utils.offload",
                    ):
                        module = sys.modules.get(module_name)
                        if module is not None and hasattr(module, "safe_open"):
                            module.safe_open = safe_open_with_pread
                except ImportError:
                    pass
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Santali speech generation is not installed. In the project folder, "
                "run `python -m pip install -r requirements-tts.txt`, then restart the server."
            ) from exc

        token = os.environ.get("HF_TOKEN") or True
        try:
            load_started = time.perf_counter()
            print(f"[Santali TTS] loading {MODEL_ID} on {device}", file=sys.stderr, flush=True)
            # Avoid holding a randomly initialized model and a second full
            # checkpoint copy at once. This matters on CPU-only school servers.
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                MODEL_ID, token=token, low_cpu_mem_usage=True,
            ).to(device)
            print("[Santali TTS] model weights loaded", file=sys.stderr, flush=True)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
            description_tokenizer = AutoTokenizer.from_pretrained(
                model.config.text_encoder._name_or_path, token=token,
            )
            description_inputs = description_tokenizer(MODEL_CAPTION, return_tensors="pt").to(device)
            print("[Santali TTS] tokenizers loaded", file=sys.stderr, flush=True)
            print(
                f"[Santali TTS] model and tokenizers ready in {time.perf_counter() - load_started:.1f}s",
                file=sys.stderr, flush=True,
            )
        except Exception as exc:
            message = str(exc).casefold()
            if "1455" in message or "paging file is too small" in message:
                raise RuntimeError(
                    "Windows does not have enough committed memory for the Santali speech model (error 1455). "
                    "Bhasha Setu unloads its translation model before speech generation. Close memory-heavy apps; "
                    "if this persists, increase the Windows paging file and restart the computer. Details: "
                    + (str(exc).strip()[:180] or type(exc).__name__)
                ) from exc
            if any(word in message for word in ("gated", "401", "403", "access", "unauthorized", "forbidden")):
                raise RuntimeError(
                    "Hugging Face has not granted this server access to the Santali speech model. "
                    "Open https://huggingface.co/ai4bharat/indic-parler-tts, sign in and accept its access terms, "
                    "then set HF_TOKEN in the server's PowerShell session and retry. The model downloads on first use."
                ) from exc
            raise RuntimeError(
                "The Santali speech model could not be loaded. Check the server connection for the first download, "
                "confirm HF_TOKEN has model access, and check available memory. Details: "
                + (str(exc).strip()[:240] or type(exc).__name__)
            ) from exc

        _model, _tokenizer = model, tokenizer
        _description_tokenizer = description_tokenizer
        _description_inputs = description_inputs
        _torch, _device = torch, device
        return _model, _tokenizer, _description_tokenizer, _torch, _device


def _text_chunks(text: str, max_chars: int = 260) -> list[str]:
    sentences = re.split(r"(?<=[.!?।॥])\s+", " ".join(text.split()))
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            if current and len(current) + len(word) + 1 > max_chars:
                chunks.append(current)
                current = ""
            current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    return chunks


def generate_wav(text: str) -> bytes:
    """Generate Santali speech for the supplied, reviewer-approved text."""
    text = " ".join(str(text or "").split())
    if not text:
        raise ValueError("The approved Santali lesson text is empty.")
    if len(text) > 2000:
        raise ValueError("Keep each Santali audio lesson to 2,000 characters or fewer.")

    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Santali speech generation needs soundfile and numpy. Install requirements-tts.txt and restart the server."
        ) from exc

    was_loaded = _model is not None
    model_load_started = time.perf_counter()
    model, tokenizer, _description_tokenizer, torch, device = _load_model()
    model_load_seconds = 0.0 if was_loaded else time.perf_counter() - model_load_started
    description = _description_inputs
    sampling_rate = int(model.config.sampling_rate)
    pieces = []
    silence = np.zeros(round(sampling_rate * 0.2), dtype=np.float32)
    synthesis_started = time.perf_counter()
    with _generation_lock, torch.inference_mode():
        for chunk in _text_chunks(text):
            print("[Santali TTS] generating audio chunk", file=sys.stderr, flush=True)
            prompt = tokenizer(chunk, return_tensors="pt").to(device)
            audio = model.generate(
                input_ids=description.input_ids,
                attention_mask=description.attention_mask,
                prompt_input_ids=prompt.input_ids,
                prompt_attention_mask=prompt.attention_mask,
            ).detach().cpu().numpy().squeeze()
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
                raise RuntimeError("The speech model returned an empty or invalid audio segment.")
            pieces.extend((audio, silence))
            print("[Santali TTS] audio chunk generated", file=sys.stderr, flush=True)
    if pieces:
        pieces.pop()
    combined = np.concatenate(pieces)
    output = BytesIO()
    sf.write(output, combined, sampling_rate, format="WAV", subtype="PCM_16")
    global _last_generation_metrics
    _last_generation_metrics = {
        "model_load_seconds": round(model_load_seconds, 1),
        "synthesis_seconds": round(time.perf_counter() - synthesis_started, 1),
        "characters": len(text),
    }
    print(
        f"[Santali TTS] generated {len(text)} characters in {time.perf_counter() - synthesis_started:.1f}s",
        file=sys.stderr, flush=True,
    )
    return output.getvalue()


def _worker_main() -> None:
    """Serve JSONL requests from api.py; keep model output off the protocol stream."""
    protocol_fd = os.dup(sys.stdout.fileno())
    # Route native-library stdout to stderr too, preserving a clean pipe for
    # the one-line JSON protocol consumed by the API process.
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    for raw_line in sys.stdin.buffer:
        try:
            request = json.loads(raw_line.decode("utf-8"))
            text = str(request.get("text") or "")
            output_path = Path(str(request.get("output_path") or ""))
            if not output_path.name:
                raise ValueError("No temporary audio path was provided.")
            # Transformers and progress helpers may print to stdout. Redirect that
            # output so api.py receives exactly one JSON response per request.
            with redirect_stdout(sys.stderr):
                audio = generate_wav(text)
            output_path.write_bytes(audio)
            result = {"ok": True, "timings": _last_generation_metrics}
        except BaseException as exc:
            # Some ML libraries terminate a worker with SystemExit instead of
            # raising Exception. Convert that into a normal job error so the
            # server can keep running and report the actual cause to reviewers.
            import traceback
            traceback.print_exc(limit=8, file=sys.stderr)
            error = f"{type(exc).__name__}: {str(exc).strip()}".strip()
            result = {"ok": False, "error": error[:400] or type(exc).__name__}
        try:
            os.write(protocol_fd, (json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8"))
        except (BrokenPipeError, OSError):
            return


if __name__ == "__main__" and "--worker" in sys.argv[1:]:
    _worker_main()
