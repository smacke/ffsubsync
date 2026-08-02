"""Browser bridge for ffsubsync.

Loaded into the Pyodide runtime; exposes JSON-friendly entry points the JS worker
calls. Two sync paths:

* ``sync_subtitles`` — subtitle-vs-subtitle (Phase 1). Pure numpy + subtitle
  parsing; no ffmpeg / VAD.
* ``sync_with_audio`` — video/audio reference (Phase 2). The browser decodes audio
  to PCM with ffmpeg.wasm and hands us the raw samples; we run ffsubsync's own VAD
  detector over them to build the reference speech signal, serialize it, and rejoin
  the standard sync path.

Both delegate the actual alignment to the *real*, vendored ffsubsync code so results
match the CLI. Input/output files live in Pyodide's in-memory MEMFS.
"""

import os
import traceback

import numpy as np

from ffsubsync.constants import SAMPLE_RATE
from ffsubsync.ffsubsync import make_parser, run
from ffsubsync.speech_transformers import (
    _make_auditok_detector,
    _make_webrtcvad_detector,
    _parse_auditok_spec,
)

_WORK_DIR = "/work"


def _ensure_workdir() -> None:
    os.makedirs(_WORK_DIR, exist_ok=True)


def _as_bytes(data) -> bytes:
    """Coerce whatever the JS worker handed us into real Python bytes.

    In the browser, file contents arrive as a JS ``Uint8Array`` (a JsProxy with a
    ``.to_py()`` returning a memoryview); natively they are already bytes. Handle
    both so the same bridge works in tests and in Pyodide.
    """
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    to_py = getattr(data, "to_py", None)
    if to_py is not None:
        data = to_py()
    return bytes(data)


def _write(name: str, data) -> str:
    path = os.path.join(_WORK_DIR, name)
    with open(path, "wb") as f:
        f.write(_as_bytes(data))
    return path


def _synced_name(in_name: str) -> str:
    stem, _ext = os.path.splitext(os.path.basename(in_name))
    return "{}.synced.srt".format(stem)


def _common_argv(in_path, out_path, *, output_encoding, no_fix_framerate, gss,
                 max_offset_seconds, split_sync=False):
    argv = ["-i", in_path, "-o", out_path]
    if output_encoding:
        argv += ["--output-encoding", output_encoding]
    if no_fix_framerate:
        argv += ["--no-fix-framerate"]
    if gss:
        argv += ["--gss"]
    if split_sync:
        # Bare flag => alass-style piecewise sync with the built-in default penalty.
        argv += ["--split-penalty"]
    if max_offset_seconds is not None:
        argv += ["--max-offset-seconds", str(max_offset_seconds)]
    return argv


def _run_and_collect(argv, out_path, in_name):
    """Run ffsubsync with ``argv`` and package a JS-friendly result dict."""
    try:
        args = make_parser().parse_args(argv)
        result = run(args)
    except Exception:  # pragma: no cover - defensive; surfaced to the UI
        return {
            "ok": False,
            "offset_seconds": None,
            "framerate_scale_factor": None,
            "output_name": _synced_name(in_name),
            "output_text": "",
            "error": traceback.format_exc(),
        }

    ok = bool(result.get("sync_was_successful")) and os.path.exists(out_path)
    output_text = ""
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8", errors="replace") as f:
            output_text = f.read()

    return {
        "ok": ok,
        "offset_seconds": result.get("offset_seconds"),
        "framerate_scale_factor": result.get("framerate_scale_factor"),
        "output_name": _synced_name(in_name),
        "output_text": output_text,
        "error": None if ok else "sync did not succeed (see console logs)",
    }


def sync_subtitles(
    ref_name: str,
    ref_bytes,
    in_name: str,
    in_bytes,
    *,
    reference_encoding=None,
    output_encoding: str = "utf-8",
    no_fix_framerate: bool = False,
    gss: bool = False,
    split_sync: bool = False,
    max_offset_seconds=None,
):
    """Sync ``in_bytes`` against subtitle reference ``ref_bytes`` (Phase 1)."""
    _ensure_workdir()
    ref_path = _write("reference_" + os.path.basename(ref_name), ref_bytes)
    in_path = _write("input_" + os.path.basename(in_name), in_bytes)
    out_path = os.path.join(_WORK_DIR, "output.srt")

    argv = [ref_path] + _common_argv(
        in_path, out_path, output_encoding=output_encoding,
        no_fix_framerate=no_fix_framerate, gss=gss, split_sync=split_sync,
        max_offset_seconds=max_offset_seconds,
    )
    if reference_encoding:
        argv += ["--reference-encoding", reference_encoding]
    return _run_and_collect(argv, out_path, in_name)


# ---- Phase 2: video/audio reference -----------------------------------------

_DETECTOR_MAKERS = {
    "webrtc": _make_webrtcvad_detector,
    "auditok": _make_auditok_detector,
}


def _pcm_to_speech_signal(pcm: bytes, frame_rate: int, vad: str,
                          non_speech_label: float) -> np.ndarray:
    """Turn mono s16le PCM into ffsubsync's speech signal via the chosen VAD.

    This mirrors ``VideoSpeechTransformer._fit_using_audio`` exactly, minus the
    ffmpeg subprocess: the audio has already been decoded (by ffmpeg.wasm in the
    browser), so we just chunk the bytes on the same window boundary and feed each
    chunk to the same detector, then concatenate. ``frame_rate`` is the PCM sample
    rate; for webrtcvad it must be one of 8000/16000/32000/48000 Hz.
    """
    if vad.startswith("auditok"):
        # supports the parameterized forms too ("auditok:75", "auditok:pXX",
        # "auditok:webrtc[:N]"); bare "auditok" auto-estimates the threshold
        detector = _make_auditok_detector(
            SAMPLE_RATE,
            frame_rate,
            non_speech_label,
            vad_spec=_parse_auditok_spec(vad),
        )
    else:
        maker = _DETECTOR_MAKERS.get(vad)
        if maker is None:
            raise ValueError(
                "unsupported browser vad %r; expected one of %s"
                % (vad, ", ".join(sorted(_DETECTOR_MAKERS)))
            )
        detector = maker(SAMPLE_RATE, frame_rate, non_speech_label)

    bytes_per_frame = 2
    frames_per_window = bytes_per_frame * frame_rate // SAMPLE_RATE
    windows_per_buffer = 10000
    step = frames_per_window * windows_per_buffer  # bytes per chunk (window-aligned)

    parts = []
    for start in range(0, len(pcm), step):
        chunk = pcm[start:start + step]
        if not chunk:
            break
        parts.append(detector(np.frombuffer(chunk, np.uint8)))
    if not parts:
        raise ValueError("no audio samples to analyze")
    return np.concatenate(parts)


def sync_with_audio(
    ref_pcm,
    frame_rate: int,
    in_name: str,
    in_bytes,
    *,
    vad: str = "webrtc",
    non_speech_label: float = 0.0,
    output_encoding: str = "utf-8",
    no_fix_framerate: bool = False,
    gss: bool = False,
    split_sync: bool = False,
    max_offset_seconds=None,
):
    """Sync ``in_bytes`` against decoded audio PCM from a video/audio reference.

    ``ref_pcm`` is mono signed-16-bit little-endian PCM decoded at ``frame_rate``
    (the browser produces this with ffmpeg.wasm). We build the reference speech
    signal with ``vad`` ("webrtc" or "auditok"), serialize it to an ``.npz``, and
    hand that to the standard ffsubsync reference path (DeserializeSpeechTransformer).
    """
    _ensure_workdir()
    try:
        signal = _pcm_to_speech_signal(
            _as_bytes(ref_pcm), int(frame_rate), vad, non_speech_label
        )
    except Exception:
        return {
            "ok": False,
            "offset_seconds": None,
            "framerate_scale_factor": None,
            "output_name": _synced_name(in_name),
            "output_text": "",
            "error": traceback.format_exc(),
        }

    npz_path = os.path.join(_WORK_DIR, "reference.npz")
    np.savez_compressed(npz_path, speech=signal)
    in_path = _write("input_" + os.path.basename(in_name), in_bytes)
    out_path = os.path.join(_WORK_DIR, "output.srt")

    argv = [npz_path] + _common_argv(
        in_path, out_path, output_encoding=output_encoding,
        no_fix_framerate=no_fix_framerate, gss=gss, split_sync=split_sync,
        max_offset_seconds=max_offset_seconds,
    )
    argv += ["--non-speech-label", str(non_speech_label)]
    return _run_and_collect(argv, out_path, in_name)
