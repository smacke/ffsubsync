# -*- coding: utf-8 -*-
from unittest.mock import patch

import pytest

from ffsubsync.speech_transformers import _get_pgs_timings_via_ffprobe


def _make_packet(pts_time, duration_time, size):
    return {
        "pts_time": str(pts_time),
        "duration_time": "N/A" if duration_time is None else str(duration_time),
        "size": str(size),
    }


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_basic(mock_probe, mock_bin):
    mock_probe.return_value = {
        "packets": [
            _make_packet(1.0, 2.5, 1000),
            _make_packet(5.0, 1.0, 800),
        ]
    }
    result = _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0")
    assert result == [(1.0, 3.5), (5.0, 6.0)]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_strips_0_prefix_from_stream(mock_probe, mock_bin):
    """'0:s:0' should be passed to ffprobe as 's:0'."""
    mock_probe.return_value = {"packets": [_make_packet(0.0, 1.0, 100)]}
    _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0")
    _, kwargs = mock_probe.call_args
    assert kwargs["select_streams"] == "s:0"


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_stream_without_prefix_unchanged(mock_probe, mock_bin):
    mock_probe.return_value = {"packets": [_make_packet(0.0, 1.0, 100)]}
    _get_pgs_timings_via_ffprobe("test.mkv", "s:1")
    _, kwargs = mock_probe.call_args
    assert kwargs["select_streams"] == "s:1"


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_skips_clear_events_small_size(mock_probe, mock_bin):
    """Packets with size <= 50 are clear events and must be skipped."""
    mock_probe.return_value = {
        "packets": [
            _make_packet(1.0, 2.0, 1000),  # show event
            _make_packet(3.0, 0.001, 30),  # clear event, size <= 50
        ]
    }
    result = _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0")
    assert result == [(1.0, 3.0)]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_skips_na_duration(mock_probe, mock_bin):
    """Packets with duration_time=N/A must be skipped."""
    mock_probe.return_value = {
        "packets": [
            _make_packet(1.0, None, 1000),  # N/A duration
            _make_packet(5.0, 2.0, 900),
        ]
    }
    result = _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0")
    assert result == [(5.0, 7.0)]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_returns_none_when_no_usable_packets(mock_probe, mock_bin):
    """Returns None if all packets are filtered out."""
    mock_probe.return_value = {
        "packets": [
            _make_packet(1.0, None, 1000),  # N/A duration
            _make_packet(2.0, 1.0, 20),  # too small
        ]
    }
    assert _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0") is None


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_returns_none_on_empty_packets(mock_probe, mock_bin):
    mock_probe.return_value = {"packets": []}
    assert _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0") is None


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_returns_none_when_ffprobe_raises(mock_probe, mock_bin):
    mock_probe.side_effect = Exception("ffprobe not found")
    assert _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0") is None


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_skips_packets_with_missing_fields(mock_probe, mock_bin):
    """Packets missing any required field are silently skipped."""
    mock_probe.return_value = {
        "packets": [
            {"pts_time": "1.0", "duration_time": "2.0"},  # missing size
            {"pts_time": "3.0", "size": "500"},  # missing duration_time
            {"duration_time": "1.0", "size": "500"},  # missing pts_time
            _make_packet(10.0, 1.0, 200),  # valid
        ]
    }
    result = _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0")
    assert result == [(10.0, 11.0)]
