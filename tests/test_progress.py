# -*- coding: utf-8 -*-
import numpy as np

import ffsubsync.speech_transformers as st
from ffsubsync.constants import DEFAULT_FRAME_RATE, DEFAULT_VAD, SAMPLE_RATE
from ffsubsync.ffsubsync import make_parser, make_reference_pipe
from ffsubsync.speech_transformers import ProgressInfo, VideoSpeechTransformer


# ---- ProgressInfo math ----------------------------------------------------

def test_progress_info_fraction():
    assert ProgressInfo(50.0, 200.0).fraction == 0.25


def test_progress_info_fraction_unknown_total_is_none():
    assert ProgressInfo(10.0, None).fraction is None
    assert ProgressInfo(10.0, 0.0).fraction is None


def test_progress_info_fraction_is_clamped_to_one():
    assert ProgressInfo(500.0, 200.0).fraction == 1.0


# ---- wiring ---------------------------------------------------------------

def test_make_reference_pipe_threads_progress_handler():
    args = make_parser().parse_args(["movie.mkv", "--vad", DEFAULT_VAD])

    def handler(_info):
        pass

    pipe = make_reference_pipe(args, progress_handler=handler)
    transformer = pipe.named_steps["speech_extract"]
    assert isinstance(transformer, VideoSpeechTransformer)
    assert transformer.progress_handler is handler


def test_make_reference_pipe_default_progress_handler_is_none():
    args = make_parser().parse_args(["movie.mkv", "--vad", DEFAULT_VAD])
    transformer = make_reference_pipe(args).named_steps["speech_extract"]
    assert transformer.progress_handler is None


# ---- callback firing through fit() ----------------------------------------

class _FakeStdout:
    """Yields a fixed PCM chunk a number of times, then EOF."""

    def __init__(self, chunk, times):
        self._chunk = chunk
        self._remaining = times

    def read(self, _n):
        if self._remaining <= 0:
            return b""
        self._remaining -= 1
        return self._chunk


class _FakeProcess:
    def __init__(self, stdout):
        self.stdout = stdout

    def wait(self):
        return 0


def _stub_video_fit(monkeypatch, total_duration, chunk, times):
    monkeypatch.setattr(
        st.ffmpeg,
        "probe",
        lambda *a, **k: {"format": {"duration": str(total_duration)}},
    )
    monkeypatch.setattr(
        st, "_make_webrtcvad_detector", lambda *a, **k: (lambda _b: np.zeros(1))
    )
    monkeypatch.setattr(
        st.subprocess, "Popen", lambda *a, **k: _FakeProcess(_FakeStdout(chunk, times))
    )


def _transformer(**overrides):
    kwargs = dict(
        vad="webrtc",
        sample_rate=SAMPLE_RATE,
        frame_rate=DEFAULT_FRAME_RATE,
        non_speech_label=0.0,
    )
    kwargs.update(overrides)
    return VideoSpeechTransformer(**kwargs)


def test_progress_handler_fires_with_monotonic_progress(monkeypatch):
    # 1 second of mono s16le audio == frame_rate * 2 bytes
    one_second = b"\x00" * (DEFAULT_FRAME_RATE * 2)
    _stub_video_fit(monkeypatch, total_duration=100.0, chunk=one_second, times=5)

    seen = []
    _transformer(progress_handler=seen.append).fit("ref.mkv")

    assert [p.processed_seconds for p in seen] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert all(p.total_seconds == 100.0 for p in seen)
    assert seen[-1].fraction == 0.05


def test_progress_handler_exception_does_not_break_fit(monkeypatch):
    one_second = b"\x00" * (DEFAULT_FRAME_RATE * 2)
    _stub_video_fit(monkeypatch, total_duration=10.0, chunk=one_second, times=3)

    def boom(_info):
        raise RuntimeError("boom")

    # must complete without raising despite the handler blowing up every call
    _transformer(progress_handler=boom).fit("ref.mkv")
