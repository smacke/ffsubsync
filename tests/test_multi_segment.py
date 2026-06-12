# -*- coding: utf-8 -*-
import numpy as np
import pytest

import ffsubsync.speech_transformers as st
from ffsubsync.aligners import FFTAligner
from ffsubsync.constants import FRAMERATE_RATIOS, SAMPLE_RATE
from ffsubsync.ffsubsync import make_parser, make_reference_pipe
from ffsubsync.speech_transformers import MultiSegmentVideoSpeechTransformer

SR = SAMPLE_RATE


def _transformer(**overrides):
    kwargs = dict(
        vad="webrtc",
        sample_rate=SR,
        frame_rate=48000,
        non_speech_label=0.0,
        segment_duration=60,
    )
    kwargs.update(overrides)
    return MultiSegmentVideoSpeechTransformer(**kwargs)


# ---- segment position logic ----------------------------------------------

def test_segment_starts_are_evenly_spaced_and_in_range():
    t = _transformer(segment_count=8)
    starts = t._segment_starts(600.0)
    assert len(starts) == 8
    assert starts == sorted(starts)
    assert starts[0] == 0
    assert all(0 <= s <= 600 - t.segment_duration for s in starts)


def test_segment_starts_single_when_reference_shorter_than_segment():
    t = _transformer(segment_count=8)
    assert t._segment_starts(40.0) == [0]


def test_segment_starts_respects_intro_outro_margins():
    t = _transformer(segment_count=6, skip_intro_outro=True)
    starts = t._segment_starts(900.0)
    assert starts[0] >= t.START_MARGIN_SECONDS
    assert starts[-1] <= 900 - t.END_MARGIN_SECONDS - t.segment_duration


def test_vad_prefix_is_stripped_for_audio_sampling():
    # "subs_then_webrtc" would otherwise try per-segment embedded-subtitle extraction
    assert _transformer(vad="subs_then_webrtc").vad == "webrtc"
    assert _transformer(vad="fused:union").vad == "fused:union"


# ---- sparse-signal assembly ----------------------------------------------

def _stub_probe(monkeypatch, duration):
    monkeypatch.setattr(
        st.ffmpeg, "probe", lambda *a, **k: {"format": {"duration": str(duration)}}
    )


def test_fit_assembles_sparse_signal(monkeypatch):
    _stub_probe(monkeypatch, 120.0)
    t = _transformer(segment_count=3, segment_duration=10)
    # each segment returns a constant-1 window; everything else stays zero
    monkeypatch.setattr(
        t, "_extract_segment_speech", lambda fname, start: (start, np.ones(10 * SR))
    )
    t.fit("ref.mkv")
    speech = t.transform()
    assert len(speech) == int(120 * SR) + 2
    starts = t._segment_starts(120.0)
    for s in starts:
        assert np.all(speech[s * SR : s * SR + 10 * SR] == 1.0)
    # a point well outside any sampled window is zero
    gap = starts[0] * SR + 10 * SR + 5
    if gap < starts[1] * SR:
        assert speech[gap] == 0.0


def test_fit_tolerates_partial_segment_failures(monkeypatch):
    _stub_probe(monkeypatch, 120.0)
    t = _transformer(segment_count=3, segment_duration=10)
    starts = t._segment_starts(120.0)
    failing = starts[0]

    def flaky(fname, start):
        if start == failing:
            raise RuntimeError("ffmpeg blew up")
        return start, np.ones(10 * SR)

    monkeypatch.setattr(t, "_extract_segment_speech", flaky)
    t.fit("ref.mkv")  # must not raise
    speech = t.transform()
    # the failed window is empty, the others are filled
    assert np.all(speech[failing * SR : failing * SR + 10 * SR] == 0.0)
    assert np.all(speech[starts[1] * SR : starts[1] * SR + 10 * SR] == 1.0)


def test_fit_raises_when_no_speech_detected(monkeypatch):
    _stub_probe(monkeypatch, 120.0)
    t = _transformer(segment_count=3)
    monkeypatch.setattr(
        t, "_extract_segment_speech", lambda fname, start: (start, np.zeros(60 * SR))
    )
    with pytest.raises(ValueError, match="Unable to detect speech"):
        t.fit("ref.mkv")


# ---- end-to-end: sparse signal recovers offset AND framerate -------------

def _scaled(sub, sf):
    """Emulate SubtitleScaler: subtitle timestamps multiplied by sf."""
    out = np.zeros(int(len(sub) * sf) + 2)
    k = np.arange(len(out))
    src = np.round(k / sf).astype(int)
    ok = src < len(sub)
    out[k[ok]] = sub[src[ok]]
    return out


def _best_ratio_and_offset(sparse_ref, sub):
    candidates = [1.0]
    candidates += list(FRAMERATE_RATIOS) + [1.0 / r for r in FRAMERATE_RATIOS]
    best = None
    for sf in candidates:
        aligner = FFTAligner(max_offset_samples=60 * SR)
        aligner.fit(sparse_ref, _scaled(sub, sf), get_score=True)
        score, offset = aligner.transform()
        if best is None or score > best[0]:
            best = (score, offset, sf)
    return best[2], best[1] / SR


@pytest.mark.parametrize(
    "true_scale, true_shift",
    [
        (1.0, 5.0),
        (1.0, -8.0),
        (25.0 / 24.0, 3.0),  # large framerate ratio the regression couldn't handle
        (24.0 / 25.0, -2.0),
    ],
)
def test_sparse_reference_recovers_scale_and_offset(monkeypatch, true_scale, true_shift):
    rng = np.random.RandomState(13)
    n_sub = 24000
    n_ref = int(true_scale * n_sub + abs(true_shift) * SR) + 2000
    ref_full = (rng.rand(n_ref) > 0.6).astype(float)
    m = np.arange(n_sub)
    idx = np.round(true_scale * m + true_shift * SR).astype(int)
    sub = np.zeros(n_sub)
    ok = (idx >= 0) & (idx < n_ref)
    sub[m[ok]] = ref_full[idx[ok]]

    _stub_probe(monkeypatch, n_ref / SR)
    t = _transformer(segment_count=8, segment_duration=60)
    # extraction returns the true reference VAD for the requested window
    monkeypatch.setattr(
        t,
        "_extract_segment_speech",
        lambda fname, start: (start, ref_full[start * SR : (start + t.segment_duration) * SR]),
    )
    t.fit("ref.mkv")
    scale, offset = _best_ratio_and_offset(t.transform(), sub)
    assert scale == pytest.approx(true_scale, abs=1e-3)
    assert offset == pytest.approx(true_shift, abs=0.05)


# ---- CLI / pipeline wiring -----------------------------------------------

def test_cli_parses_multi_segment_flags():
    args = make_parser().parse_args(
        [
            "movie.mkv",
            "--multi-segment-sync",
            "--segment-count", "10",
            "--skip-intro-outro",
            "--parallel-workers", "6",
        ]
    )
    assert args.multi_segment_sync is True
    assert args.segment_count == 10
    assert args.skip_intro_outro is True
    assert args.parallel_workers == 6


def test_make_reference_pipe_uses_multi_segment_transformer():
    args = make_parser().parse_args(
        ["movie.mkv", "--vad", "webrtc", "--multi-segment-sync", "--segment-count", "5"]
    )
    transformer = make_reference_pipe(args).named_steps["speech_extract"]
    assert isinstance(transformer, MultiSegmentVideoSpeechTransformer)
    assert transformer.segment_count == 5


def test_make_reference_pipe_uses_plain_transformer_without_flag():
    args = make_parser().parse_args(["movie.mkv", "--vad", "webrtc"])
    transformer = make_reference_pipe(args).named_steps["speech_extract"]
    assert not isinstance(transformer, MultiSegmentVideoSpeechTransformer)
