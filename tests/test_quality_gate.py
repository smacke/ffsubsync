# -*- coding: utf-8 -*-
import types

import numpy as np
import pytest

from ffsubsync import ffsubsync
from ffsubsync.constants import (
    DEFAULT_MAX_FRAMERATE_DEVIATION,
    DEFAULT_MIN_SCORE,
    DEFAULT_QUALITY_MAX_OFFSET_SECONDS,
    SAMPLE_RATE,
)
from ffsubsync.ffsubsync import assess_alignment_quality
from ffsubsync.subtitle_parser import GenericSubtitleParser

SRT = """1
00:00:10,000 --> 00:00:11,000
Hello

2
00:00:20,000 --> 00:00:21,000
World
"""

_THRESHOLDS = dict(
    min_score=DEFAULT_MIN_SCORE,
    max_offset_seconds=DEFAULT_QUALITY_MAX_OFFSET_SECONDS,
    max_framerate_deviation=DEFAULT_MAX_FRAMERATE_DEVIATION,
)


# ---- pure assessment logic -----------------------------------------------

def test_quality_ok_for_plausible_alignment():
    assert assess_alignment_quality(500.0, 3.0, 1.0, **_THRESHOLDS) == []


def test_quality_rejects_negative_score():
    reasons = assess_alignment_quality(-1.0, 3.0, 1.0, **_THRESHOLDS)
    assert len(reasons) == 1 and "score" in reasons[0]


def test_quality_rejects_large_offset():
    reasons = assess_alignment_quality(500.0, 45.0, 1.0, **_THRESHOLDS)
    assert len(reasons) == 1 and "offset" in reasons[0]


def test_default_framerate_threshold_allows_all_real_corrections():
    # the biggest correction ffsubsync makes is 25/23.976 ~= 1.0427, and a typical
    # --gss result stays well within +/-0.1; none should trip the default threshold
    for scale in (25.0 / 24.0, 25.0 / 23.976, 24.0 / 25.0, 1.05, 0.95):
        assert assess_alignment_quality(500.0, 3.0, scale, **_THRESHOLDS) == []


def test_tightened_framerate_threshold_rejects_correction():
    reasons = assess_alignment_quality(
        500.0,
        3.0,
        25.0 / 24.0,
        min_score=0.0,
        max_offset_seconds=30.0,
        max_framerate_deviation=0.01,
    )
    assert len(reasons) == 1 and "framerate" in reasons[0]


def test_quality_reports_multiple_reasons():
    reasons = assess_alignment_quality(
        -5.0, 99.0, 1.5, min_score=0.0, max_offset_seconds=30.0, max_framerate_deviation=0.1
    )
    assert len(reasons) == 3


# ---- end-to-end behavior in try_sync -------------------------------------

def _fake_aligner(score, offset_samples):
    class _Fake:
        def __init__(self, *a, **k):
            pass

        def fit_transform(self, refstring, subpipes):
            return (score, offset_samples), subpipes[0]

    return _Fake


def _run_try_sync(tmp_path, monkeypatch, *, score, offset_seconds, **arg_overrides):
    srtin = tmp_path / "in.srt"
    srtin.write_text(SRT)
    srtout = tmp_path / "out.srt"
    args = ffsubsync.make_parser().parse_args(
        ["ref.mkv", "-i", str(srtin), "-o", str(srtout)]
    )
    args.skip_infer_framerate_ratio = True
    for k, v in arg_overrides.items():
        setattr(args, k, v)
    monkeypatch.setattr(
        ffsubsync,
        "MaxScoreAligner",
        _fake_aligner(score, int(offset_seconds * SAMPLE_RATE)),
    )
    reference_pipe = types.SimpleNamespace(transform=lambda _: np.zeros(10))
    result = {"retval": 0}
    ok = ffsubsync.try_sync(args, reference_pipe, result)
    return ok, srtout


def _first_start_seconds(path):
    parser = GenericSubtitleParser()
    parser.fit(str(path))
    return list(parser.subs_)[0].start.total_seconds()


def test_try_sync_skips_and_writes_original_on_low_quality(tmp_path, monkeypatch):
    # offset of 300s blows past the 30s quality threshold -> leave subs unmodified
    ok, srtout = _run_try_sync(
        tmp_path, monkeypatch, score=500.0, offset_seconds=300.0,
        skip_sync_on_low_quality=True, max_offset_seconds=600,
    )
    assert ok is False
    assert _first_start_seconds(srtout) == pytest.approx(10.0)  # unchanged


def test_try_sync_applies_when_quality_is_acceptable(tmp_path, monkeypatch):
    ok, srtout = _run_try_sync(
        tmp_path, monkeypatch, score=500.0, offset_seconds=5.0,
        skip_sync_on_low_quality=True,
    )
    assert ok is True
    assert _first_start_seconds(srtout) == pytest.approx(15.0)  # shifted +5s


def test_try_sync_ignores_quality_without_flag(tmp_path, monkeypatch):
    # same bad offset, but the flag is off, so it is applied anyway
    ok, srtout = _run_try_sync(
        tmp_path, monkeypatch, score=500.0, offset_seconds=300.0,
        skip_sync_on_low_quality=False, max_offset_seconds=600,
    )
    assert ok is True
    assert _first_start_seconds(srtout) == pytest.approx(310.0)


def test_cli_parses_quality_flags():
    args = ffsubsync.make_parser().parse_args(
        [
            "movie.mkv",
            "--skip-sync-on-low-quality",
            "--min-score", "100",
            "--quality-max-offset-seconds", "20",
            "--max-framerate-deviation", "0.03",
        ]
    )
    assert args.skip_sync_on_low_quality is True
    assert args.min_score == 100.0
    assert args.quality_max_offset_seconds == 20.0
    assert args.max_framerate_deviation == 0.03
