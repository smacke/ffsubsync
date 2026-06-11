# -*- coding: utf-8 -*-
import os
from datetime import timedelta

from ffsubsync.constants import DEFAULT_FRAME_RATE, DEFAULT_VAD, SAMPLE_RATE
from ffsubsync.ffsubsync import make_parser, make_reference_pipe
from ffsubsync.speech_transformers import VideoSpeechTransformer

REMOTE_URL = "https://example.com/video.mp4"


def _transformer(**overrides):
    kwargs = dict(
        vad="webrtc",  # avoid the embedded-subs branch
        sample_rate=SAMPLE_RATE,
        frame_rate=DEFAULT_FRAME_RATE,
        non_speech_label=0.0,
        extract_audio_first=True,
    )
    kwargs.update(overrides)
    return VideoSpeechTransformer(**kwargs)


def _fake_ffmpeg_success(monkeypatch):
    """subprocess.call that writes non-empty data to the output path and succeeds."""
    captured = {}

    def fake_call(args, **kwargs):
        captured["args"] = args
        with open(args[-1], "wb") as f:
            f.write(b"\x00" * 16)
        return 0

    monkeypatch.setattr(
        "ffsubsync.speech_transformers.subprocess.call", fake_call
    )
    return captured


def test_extract_audio_builds_copy_args_and_returns_temp(monkeypatch):
    captured = _fake_ffmpeg_success(monkeypatch)
    transformer = _transformer(max_duration_seconds=None)
    temp_path = transformer._extract_audio_to_temp(REMOTE_URL)
    try:
        assert temp_path is not None and temp_path.endswith(".mka")
        assert os.path.exists(temp_path)
        args = captured["args"]
        assert args[args.index("-i") + 1] == REMOTE_URL
        assert "-vn" in args  # drop video
        assert args[args.index("-acodec") + 1] == "copy"
        assert "-t" not in args  # no duration limit requested
        assert args[-1] == temp_path
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def test_extract_audio_t_limit_includes_start_offset(monkeypatch):
    captured = _fake_ffmpeg_success(monkeypatch)
    transformer = _transformer(start_seconds=30, max_duration_seconds=600)
    temp_path = transformer._extract_audio_to_temp(REMOTE_URL)
    try:
        args = captured["args"]
        # extract t=0..(start+max) so the main pass can still seek start accurately
        assert args[args.index("-t") + 1] == str(timedelta(seconds=630))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def test_extract_audio_returns_none_and_cleans_up_on_failure(monkeypatch):
    leaked = {}

    def fake_call(args, **kwargs):
        leaked["path"] = args[-1]  # left empty (size 0) -> treated as failure
        return 1

    monkeypatch.setattr(
        "ffsubsync.speech_transformers.subprocess.call", fake_call
    )
    assert _transformer()._extract_audio_to_temp(REMOTE_URL) is None
    assert not os.path.exists(leaked["path"])


def test_fit_extracts_then_cleans_up_temp(monkeypatch, tmp_path):
    temp_file = tmp_path / "audio.mka"
    temp_file.write_bytes(b"\x00" * 16)
    seen = {}

    transformer = _transformer()
    monkeypatch.setattr(
        transformer, "_extract_audio_to_temp", lambda url: str(temp_file)
    )
    monkeypatch.setattr(
        transformer, "_fit_using_audio", lambda fname: seen.update(fname=fname)
    )
    transformer.fit(REMOTE_URL)
    # the local temp path was used for detection, then removed afterward
    assert seen["fname"] == str(temp_file)
    assert not temp_file.exists()


def test_fit_skips_extraction_for_local_reference(monkeypatch):
    seen = {}
    transformer = _transformer()

    def fail(url):
        raise AssertionError("should not extract audio for a local reference")

    monkeypatch.setattr(transformer, "_extract_audio_to_temp", fail)
    monkeypatch.setattr(
        transformer, "_fit_using_audio", lambda fname: seen.update(fname=fname)
    )
    transformer.fit("/local/movie.mkv")
    assert seen["fname"] == "/local/movie.mkv"


def test_fit_skips_extraction_when_flag_disabled(monkeypatch):
    seen = {}
    transformer = _transformer(extract_audio_first=False)

    def fail(url):
        raise AssertionError("should not extract audio when flag disabled")

    monkeypatch.setattr(transformer, "_extract_audio_to_temp", fail)
    monkeypatch.setattr(
        transformer, "_fit_using_audio", lambda fname: seen.update(fname=fname)
    )
    transformer.fit(REMOTE_URL)
    assert seen["fname"] == REMOTE_URL


def test_cli_and_pipeline_wiring():
    args = make_parser().parse_args(
        ["movie.mkv", "--vad", DEFAULT_VAD, "--extract-audio-first"]
    )
    assert args.extract_audio_first is True
    assert make_parser().parse_args(["movie.mkv"]).extract_audio_first is False
    transformer = make_reference_pipe(args).named_steps["speech_extract"]
    assert transformer.extract_audio_first is True
