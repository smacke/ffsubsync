# -*- coding: utf-8 -*-
import builtins

import numpy as np
import pytest

import ffsubsync.speech_transformers as st
from ffsubsync.ffsubsync import make_parser


def _stub_factories(monkeypatch, webrtc_result, silero_result):
    """Replace the webrtc/silero detector factories with stubs returning fixed arrays."""
    monkeypatch.setattr(
        st, "_make_webrtcvad_detector", lambda *a, **k: (lambda seg: np.asarray(webrtc_result, dtype=float))
    )
    monkeypatch.setattr(
        st, "_make_silero_detector", lambda *a, **k: (lambda seg: np.asarray(silero_result, dtype=float))
    )


def test_fused_intersection_is_elementwise_min(monkeypatch):
    _stub_factories(monkeypatch, [1.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    detector = st._make_fused_detector(100, 48000, 0.0, "intersection")
    assert list(detector(b"")) == [1.0, 0.0, 0.0]


def test_fused_union_is_elementwise_max(monkeypatch):
    _stub_factories(monkeypatch, [1.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    detector = st._make_fused_detector(100, 48000, 0.0, "union")
    assert list(detector(b"")) == [1.0, 1.0, 0.0]


def test_fused_weighted_is_silero_heavy(monkeypatch):
    _stub_factories(monkeypatch, [1.0, 0.0], [0.0, 1.0])
    detector = st._make_fused_detector(100, 48000, 0.0, "weighted")
    # 0.6 * silero + 0.4 * webrtc
    assert np.allclose(detector(b""), [0.4, 0.6])


def test_fused_default_strategy_is_weighted(monkeypatch):
    _stub_factories(monkeypatch, [1.0], [0.0])
    detector = st._make_fused_detector(100, 48000, 0.0)
    assert np.allclose(detector(b""), [0.4])


def test_fused_clips_to_common_length(monkeypatch):
    _stub_factories(monkeypatch, [1.0, 1.0, 1.0], [1.0, 1.0])  # webrtc longer by one frame
    detector = st._make_fused_detector(100, 48000, 0.0, "union")
    assert len(detector(b"")) == 2


def test_fused_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="unknown fused VAD strategy"):
        st._make_fused_detector(100, 48000, 0.0, "bogus")


def test_silero_raises_clear_error_when_torch_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        st._make_silero_detector(100, 48000, 0.0)


@pytest.mark.parametrize(
    "vad", ["fused", "fused:weighted", "fused:intersection", "fused:union"]
)
def test_cli_accepts_fused_choices(vad):
    args = make_parser().parse_args(["movie.mkv", "--vad", vad])
    assert args.vad == vad


def test_cli_rejects_invalid_fused_choice():
    with pytest.raises(SystemExit):
        make_parser().parse_args(["movie.mkv", "--vad", "fused:bogus"])
