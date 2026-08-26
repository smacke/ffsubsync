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
    """An N/A-duration SHOW with no CLEAR before the next SHOW arrives is
    discarded (not paired), so only the second, numeric-duration SHOW
    produces an interval."""
    mock_probe.return_value = {
        "packets": [
            _make_packet(1.0, None, 1000),  # N/A-duration SHOW; becomes
            # pending, then discarded when...
            _make_packet(5.0, 2.0, 900),  # ...this numeric SHOW arrives
        ]
    }
    result = _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0")
    assert result == [(5.0, 7.0)]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_missing_duration_time_key_treated_same_as_na(mock_probe, mock_bin):
    """ffprobe's ``-of json`` writer (what ``ffmpeg.probe`` uses) omits the
    ``duration_time`` key entirely when it's unavailable, rather than
    emitting the string "N/A" the way its text-based writers do. A packet
    missing the key altogether must be paired exactly like an explicit
    "N/A" packet. Modeled on real packets captured from an actual PGS
    stream via ``ffprobe -of json`` (extra fields like codec_type/pts/dts
    included, as real packets carry them, to prove they're ignored)."""
    mock_probe.return_value = {
        "packets": [
            {
                "codec_type": "subtitle",
                "stream_index": 3,
                "pts": 8008,
                "pts_time": "8.008000",
                "dts": 8008,
                "dts_time": "8.008000",
                "size": "28162",
                "pos": "22770354",
                "flags": "K__",
                # no "duration_time" key at all
            },
            {
                "codec_type": "subtitle",
                "stream_index": 3,
                "pts": 12804,
                "pts_time": "12.804000",
                "dts": 12804,
                "dts_time": "12.804000",
                "size": "30",
                "pos": "39943227",
                "flags": "K__",
                # no "duration_time" key at all
            },
        ]
    }
    result = _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0")
    assert result == [(8.008, 12.804)]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_pairs_na_duration_show_with_following_clear(mock_probe, mock_bin):
    """A SHOW packet (size > 50) with duration_time=N/A is paired with the
    pts_time of the next CLEAR packet (size <= 50) that follows it."""
    mock_probe.return_value = {
        "packets": [
            _make_packet(8.008, None, 28162),  # SHOW, N/A duration
            _make_packet(12.804, None, 30),  # CLEAR, closes the show above
        ]
    }
    result = _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0")
    assert result == [(8.008, 12.804)]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_pairs_multiple_alternating_na_duration_show_clear(mock_probe, mock_bin):
    """Multiple alternating SHOW/CLEAR N/A pairs each produce their own
    correctly-ordered interval."""
    mock_probe.return_value = {
        "packets": [
            _make_packet(8.008, None, 28162),
            _make_packet(12.804, None, 30),
            _make_packet(14.723, None, 9212),
            _make_packet(21.647, None, 30),
        ]
    }
    result = _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0")
    assert result == [(8.008, 12.804), (14.723, 21.647)]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_unmatched_trailing_na_duration_show_is_dropped(mock_probe, mock_bin):
    """A trailing N/A-duration SHOW with no CLEAR ever following it is
    dropped; it must not corrupt or suppress interval(s) already found."""
    mock_probe.return_value = {
        "packets": [
            _make_packet(8.008, None, 28162),
            _make_packet(12.804, None, 30),
            _make_packet(14.723, None, 9212),  # never closed by a CLEAR
        ]
    }
    result = _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0")
    assert result == [(8.008, 12.804)]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_returns_none_for_lone_unmatched_na_duration_show(mock_probe, mock_bin):
    """A single N/A-duration SHOW packet with no CLEAR anywhere in the
    stream produces no usable interval."""
    mock_probe.return_value = {"packets": [_make_packet(1.0, None, 1000)]}
    assert _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0") is None


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_second_of_two_consecutive_na_duration_shows_wins(mock_probe, mock_bin):
    """When two N/A-duration SHOW packets appear back-to-back with no CLEAR
    between them, the first is discarded (not carried forward or guessed)
    and only the second pairs with the CLEAR that eventually follows."""
    mock_probe.return_value = {
        "packets": [
            _make_packet(1.0, None, 1000),  # discarded: superseded below
            _make_packet(2.0, None, 1200),  # this one pairs with the clear
            _make_packet(3.0, None, 30),  # CLEAR
        ]
    }
    result = _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0")
    assert result == [(2.0, 3.0)]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_mixed_numeric_and_na_duration_packets(mock_probe, mock_bin):
    """A single packet list may mix the legacy numeric-duration SHOW
    representation with the N/A-duration SHOW+CLEAR pairing representation;
    both must produce correct, correctly-ordered intervals."""
    mock_probe.return_value = {
        "packets": [
            _make_packet(1.0, 2.5, 1000),  # numeric SHOW -> (1.0, 3.5)
            _make_packet(8.008, None, 28162),  # N/A SHOW
            _make_packet(12.804, None, 30),  # CLEAR -> (8.008, 12.804)
            _make_packet(20.0, 1.5, 900),  # numeric SHOW -> (20.0, 21.5)
        ]
    }
    result = _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0")
    assert result == [(1.0, 3.5), (8.008, 12.804), (20.0, 21.5)]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_returns_none_when_no_usable_packets(mock_probe, mock_bin):
    """Returns None if no packet pairing ever produces a usable interval --
    e.g. every packet is CLEAR-sized, so no SHOW is ever pending to close."""
    mock_probe.return_value = {
        "packets": [
            _make_packet(1.0, None, 20),  # CLEAR-sized; no pending show
            _make_packet(2.0, 1.0, 25),  # CLEAR-sized; no pending show
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
    """Packets missing pts_time or size (truly required -- unlike
    duration_time, which may legitimately be absent, see
    test_missing_duration_time_key_treated_same_as_na) are silently
    skipped."""
    mock_probe.return_value = {
        "packets": [
            {"duration_time": "2.0", "size": "1000"},  # missing pts_time
            {"pts_time": "1.0", "duration_time": "2.0"},  # missing size
            _make_packet(10.0, 1.0, 200),  # valid
        ]
    }
    result = _get_pgs_timings_via_ffprobe("test.mkv", "0:s:0")
    assert result == [(10.0, 11.0)]
