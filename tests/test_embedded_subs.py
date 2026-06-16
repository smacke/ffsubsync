# -*- coding: utf-8 -*-
"""Tests for embedded-subtitle extraction in ``VideoSpeechTransformer``.

These exercise the single-pass extraction path (one ffmpeg invocation for all
streams, enumerated via ffprobe) and its fallbacks, while guaranteeing the
extraction stays in memory -- nothing is written next to the source media.
``subprocess.Popen`` is stubbed to stand in for ffprobe/ffmpeg, but the real
subtitle-speech pipeline parses the SRT bytes, so selection logic is tested
end to end.
"""
import io
import os

import numpy as np
import pytest

import ffsubsync.speech_transformers as st
from ffsubsync.constants import DEFAULT_FRAME_RATE, SAMPLE_RATE
from ffsubsync.speech_transformers import VideoSpeechTransformer


def _srt(end_sec):
    """A valid SRT string with one short cue per second up to ``end_sec``."""
    lines = []
    for i in range(1, end_sec + 1):
        lines.append(str(i))
        lines.append(
            "00:00:{:02d},000 --> 00:00:{:02d},500".format(i - 1, i - 1)
        )
        lines.append("line {}".format(i))
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def _parse_speech_results(srt_bytes, start_seconds=0):
    """Run the real pipeline on SRT bytes to get the expected speech results."""
    pipe = st.make_subtitle_speech_pipeline(start_seconds=start_seconds).fit(
        io.BytesIO(srt_bytes)
    )
    return pipe.steps[-1][1].subtitle_speech_results_


def _transformer(**overrides):
    kwargs = dict(
        vad="subs_then_webrtc",
        sample_rate=SAMPLE_RATE,
        frame_rate=DEFAULT_FRAME_RATE,
        non_speech_label=0.0,
    )
    kwargs.update(overrides)
    return VideoSpeechTransformer(**kwargs)


class _FakeProc:
    def __init__(self, returncode=0, stdout_bytes=b""):
        self.returncode = returncode
        self._stdout = stdout_bytes

    def communicate(self, *a, **k):
        return (self._stdout, b"")


def _install_fake_popen(
    monkeypatch,
    *,
    probe_lines,
    srt_by_stream,
    probe_unavailable=False,
    probe_rc=0,
    single_pass_rc=0,
):
    """Stub ``subprocess.Popen`` to emulate ffprobe + ffmpeg.

    ``probe_lines`` is the CSV ffprobe emits (``index,codec_name`` per line).
    ``srt_by_stream`` maps an ffmpeg ``-map`` specifier (e.g. ``"0:2"``) to the
    SRT bytes that stream yields. Records every invocation in the returned
    ``calls`` list.
    """
    calls = []
    real_popen = st.subprocess.Popen

    def fake_popen(args, **kwargs):
        binary = os.path.basename(str(args[0])).lower()
        if "ffmpeg" not in binary and "ffprobe" not in binary:
            # not one of our calls (e.g. platform.system() shelling out to
            # ``ver`` on Windows) -- let it run for real
            return real_popen(args, **kwargs)
        calls.append(list(args))
        if "ffprobe" in binary:
            if probe_unavailable:
                raise OSError("ffprobe not found")
            payload = "\n".join(probe_lines).encode("utf-8")
            return _FakeProc(returncode=probe_rc, stdout_bytes=payload)
        # ffmpeg
        if args[-1] == "-":
            # per-stream extraction to stdout
            stream = args[args.index("-map") + 1]
            data = srt_by_stream.get(stream, b"")
            # ffmpeg returns nonzero when the mapped stream does not exist
            return _FakeProc(returncode=0 if data else 1, stdout_bytes=data)
        # single-pass extraction: write each stream's SRT to its output path
        if single_pass_rc == 0:
            i = 0
            while i < len(args):
                if args[i] == "-map":
                    stream = args[i + 1]
                    out_path = args[i + 4]  # -map S -f srt PATH
                    data = srt_by_stream.get(stream, b"")
                    if data:
                        with open(out_path, "wb") as f:
                            f.write(data)
                i += 1
        return _FakeProc(returncode=single_pass_rc)

    monkeypatch.setattr(st.subprocess, "Popen", fake_popen)
    return calls


def _count_ffmpeg_invocations(calls):
    return sum(1 for c in calls if "ffmpeg" in os.path.basename(c[0]).lower())


def test_single_pass_extracts_all_streams_in_one_invocation(monkeypatch):
    srt_by_stream = {"0:2": _srt(10), "0:3": _srt(30), "0:4": _srt(20)}
    calls = _install_fake_popen(
        monkeypatch,
        probe_lines=["2,subrip", "3,ass", "4,mov_text"],
        srt_by_stream=srt_by_stream,
    )
    t = _transformer()
    t.try_fit_using_embedded_subs("movie.mkv")

    # exactly one ffmpeg pass (the perf win) plus the single ffprobe call
    assert _count_ffmpeg_invocations(calls) == 1
    assert sum(1 for c in calls if "ffprobe" in os.path.basename(c[0]).lower()) == 1
    # the longest stream (0:3, 30s) is the one whose results are kept
    expected = _parse_speech_results(srt_by_stream["0:3"])
    assert t.video_speech_results_ is not None
    assert np.array_equal(t.video_speech_results_, expected)


def test_bitmap_subtitle_streams_are_skipped(monkeypatch):
    # a PGS (bitmap) stream must not be mapped -- it cannot become SRT and would
    # otherwise abort the whole single-pass extraction
    calls = _install_fake_popen(
        monkeypatch,
        probe_lines=["0,hdmv_pgs_subtitle", "1,subrip"],
        srt_by_stream={"0:1": _srt(12)},
    )
    t = _transformer()
    t.try_fit_using_embedded_subs("movie.mkv")

    ffmpeg_call = next(
        c for c in calls if "ffmpeg" in os.path.basename(c[0]).lower()
    )
    mapped = [ffmpeg_call[i + 1] for i, a in enumerate(ffmpeg_call) if a == "-map"]
    assert mapped == ["0:1"]  # only the text stream
    assert t.video_speech_results_ is not None


def test_falls_back_to_per_stream_when_ffprobe_unavailable(monkeypatch):
    # without ffprobe we cannot enumerate, so fall back to probing 0:s:0.. one
    # at a time (original behavior), stopping at the first missing stream
    srt_by_stream = {"0:s:0": _srt(8), "0:s:1": _srt(25)}
    calls = _install_fake_popen(
        monkeypatch,
        probe_lines=[],
        srt_by_stream=srt_by_stream,
        probe_unavailable=True,
    )
    t = _transformer()
    t.try_fit_using_embedded_subs("movie.mkv")

    # per-stream means multiple ffmpeg invocations, all to stdout ("-")
    ffmpeg_calls = [c for c in calls if "ffmpeg" in os.path.basename(c[0]).lower()]
    assert len(ffmpeg_calls) >= 2
    assert all(c[-1] == "-" for c in ffmpeg_calls)
    expected = _parse_speech_results(srt_by_stream["0:s:1"])
    assert np.array_equal(t.video_speech_results_, expected)


def test_single_pass_failure_falls_back_to_per_stream(monkeypatch):
    # if the combined ffmpeg pass fails wholesale, degrade to per-stream over
    # the same enumerated streams rather than giving up
    srt_by_stream = {"0:2": _srt(15)}
    calls = _install_fake_popen(
        monkeypatch,
        probe_lines=["2,subrip"],
        srt_by_stream=srt_by_stream,
        single_pass_rc=1,
    )
    t = _transformer()
    t.try_fit_using_embedded_subs("movie.mkv")

    ffmpeg_calls = [c for c in calls if "ffmpeg" in os.path.basename(c[0]).lower()]
    # one failed single pass + at least one per-stream retry
    assert len(ffmpeg_calls) >= 2
    assert t.video_speech_results_ is not None


def test_explicit_ref_stream_uses_per_stream(monkeypatch):
    calls = _install_fake_popen(
        monkeypatch,
        probe_lines=["irrelevant"],
        srt_by_stream={"0:s:3": _srt(18)},
    )
    t = _transformer(ref_stream="0:s:3")
    t.try_fit_using_embedded_subs("movie.mkv")

    # ref_stream short-circuits enumeration: no ffprobe, single mapped stream
    assert all("ffprobe" not in os.path.basename(c[0]).lower() for c in calls)
    ffmpeg_calls = [c for c in calls if "ffmpeg" in os.path.basename(c[0]).lower()]
    assert len(ffmpeg_calls) == 1
    assert ffmpeg_calls[0][ffmpeg_calls[0].index("-map") + 1] == "0:s:3"


def test_raises_when_no_subtitle_streams(monkeypatch):
    _install_fake_popen(
        monkeypatch,
        probe_lines=[],  # ffprobe finds nothing -> fall back -> per-stream all fail
        srt_by_stream={},
    )
    t = _transformer()
    with pytest.raises(ValueError, match="lack subtitle stream"):
        t.try_fit_using_embedded_subs("movie.mkv")


def test_raises_when_explicit_ref_stream_missing(monkeypatch):
    _install_fake_popen(
        monkeypatch,
        probe_lines=[],
        srt_by_stream={},
    )
    t = _transformer(ref_stream="0:s:9")
    with pytest.raises(ValueError, match="Stream 0:s:9 not found"):
        t.try_fit_using_embedded_subs("movie.mkv")


def test_extraction_writes_nothing_next_to_video(monkeypatch, tmp_path):
    # the key regression guard: extraction must not create a Subs/ folder or any
    # files alongside the source media
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"not really a video")
    _install_fake_popen(
        monkeypatch,
        probe_lines=["2,subrip", "3,ass"],
        srt_by_stream={"0:2": _srt(10), "0:3": _srt(20)},
    )
    t = _transformer()
    t.try_fit_using_embedded_subs(str(video))

    leftovers = sorted(p.name for p in tmp_path.iterdir())
    assert leftovers == ["movie.mkv"]  # nothing else created
    assert not (tmp_path / "Subs").exists()


def test_empty_extracted_stream_is_skipped(monkeypatch):
    # a stream that yields no data (empty temp file) must be skipped, not crash
    srt_by_stream = {"0:2": b"", "0:3": _srt(14)}
    _install_fake_popen(
        monkeypatch,
        probe_lines=["2,subrip", "3,subrip"],
        srt_by_stream=srt_by_stream,
    )
    t = _transformer()
    t.try_fit_using_embedded_subs("movie.mkv")

    expected = _parse_speech_results(srt_by_stream["0:3"])
    assert np.array_equal(t.video_speech_results_, expected)


def test_build_ffmpeg_args_keeps_aresample_filter():
    # regression guard: the audio-extraction path must keep the async resample
    # filter (deliberately added upstream to fix audio timestamp drift)
    t = _transformer(vad="webrtc")
    args = t._build_ffmpeg_args("movie.mkv")
    assert "-af" in args
    assert args[args.index("-af") + 1] == "aresample=async=1"
