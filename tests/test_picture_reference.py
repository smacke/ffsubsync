# -*- coding: utf-8 -*-
from unittest.mock import patch

from ffsubsync.speech_transformers import (
    _get_dvd_timings_via_ffprobe,
    find_pgs_stream,
)


def _stream(index, codec, forced=0, comment=0, title=None, frames=None):
    tags = {}
    if title is not None:
        tags["title"] = title
    if frames is not None:
        tags["NUMBER_OF_FRAMES-eng"] = str(frames)
    return {
        "index": index,
        "codec_type": "subtitle",
        "codec_name": codec,
        "disposition": {"forced": forced, "comment": comment},
        "tags": tags,
    }


def _frame(pts_time, end_ms, num_rects=1, start_ms=0):
    return {
        "pts_time": "%.6f" % pts_time,
        "start_display_time": start_ms,
        "end_display_time": end_ms,
        "num_rects": num_rects,
    }


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_find_counts_only_subtitle_streams(mock_probe, mock_bin):
    mock_probe.return_value = {
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264"},
            _stream(1, "subrip"),
            _stream(2, "hdmv_pgs_subtitle"),
        ]
    }
    assert find_pgs_stream("test.mkv") == "0:s:1"


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_find_dvd_and_dvb(mock_probe, mock_bin):
    for codec in ("dvd_subtitle", "dvb_subtitle"):
        mock_probe.return_value = {"streams": [_stream(0, "subrip"), _stream(1, codec)]}
        assert find_pgs_stream("test.mkv") == "0:s:1"


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_find_none_without_image_subtitles(mock_probe, mock_bin):
    mock_probe.return_value = {"streams": [_stream(0, "subrip"), _stream(1, "ass")]}
    assert find_pgs_stream("test.mkv") is None


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_find_passes_over_forced(mock_probe, mock_bin):
    mock_probe.return_value = {
        "streams": [
            _stream(2, "hdmv_pgs_subtitle", forced=1),
            _stream(3, "hdmv_pgs_subtitle"),
        ]
    }
    assert find_pgs_stream("test.mkv") == "0:s:1"


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_find_uses_forced_when_it_is_all_there_is(mock_probe, mock_bin):
    mock_probe.return_value = {"streams": [_stream(2, "hdmv_pgs_subtitle", forced=1)]}
    assert find_pgs_stream("test.mkv") == "0:s:0"


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_find_passes_over_commentary(mock_probe, mock_bin):
    mock_probe.return_value = {
        "streams": [
            _stream(2, "hdmv_pgs_subtitle", title="Director's Commentary", frames=900),
            _stream(3, "hdmv_pgs_subtitle", comment=1, frames=900),
            _stream(4, "hdmv_pgs_subtitle", frames=500),
        ]
    }
    assert find_pgs_stream("test.mkv") == "0:s:2"


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_find_uses_commentary_when_it_is_all_there_is(mock_probe, mock_bin):
    mock_probe.return_value = {
        "streams": [_stream(2, "hdmv_pgs_subtitle", title="Commentary")]
    }
    assert find_pgs_stream("test.mkv") == "0:s:0"


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_find_prefers_most_cues(mock_probe, mock_bin):
    mock_probe.return_value = {
        "streams": [
            _stream(2, "hdmv_pgs_subtitle", frames=40),
            _stream(3, "hdmv_pgs_subtitle", frames=1500),
            _stream(4, "hdmv_pgs_subtitle", frames=1200),
        ]
    }
    assert find_pgs_stream("test.mkv") == "0:s:1"


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_find_keeps_first_of_equals(mock_probe, mock_bin):
    mock_probe.return_value = {
        "streams": [
            _stream(2, "hdmv_pgs_subtitle"),
            _stream(3, "hdmv_pgs_subtitle"),
        ]
    }
    assert find_pgs_stream("test.mkv") == "0:s:0"


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_dvd_timings_from_end_display_time(mock_probe, mock_bin):
    mock_probe.return_value = {"frames": [_frame(7.0, 2992), _frame(14.0, 1500)]}
    assert _get_dvd_timings_via_ffprobe("test.mkv", "0:s:0") == [
        (7.0, 9.992),
        (14.0, 15.5),
    ]
    _, kwargs = mock_probe.call_args
    assert kwargs["select_streams"] == "s:0"
    assert "show_frames" in kwargs


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_dvd_timings_honour_start_display_time(mock_probe, mock_bin):
    mock_probe.return_value = {"frames": [_frame(7.0, 3000, start_ms=500)]}
    assert _get_dvd_timings_via_ffprobe("test.mkv", "0:s:0") == [(7.5, 10.0)]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_dvd_timings_end_at_next_caption(mock_probe, mock_bin):
    mock_probe.return_value = {"frames": [_frame(7.0, 5000), _frame(9.0, 1000)]}
    assert _get_dvd_timings_via_ffprobe("test.mkv", "0:s:0") == [
        (7.0, 9.0),
        (9.0, 10.0),
    ]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_dvd_timings_empty_frame_clears(mock_probe, mock_bin):
    mock_probe.return_value = {
        "frames": [_frame(7.0, 30000), _frame(10.0, 30000, num_rects=0)]
    }
    assert _get_dvd_timings_via_ffprobe("test.mkv", "0:s:0") == [(7.0, 10.0)]


@patch("ffsubsync.speech_transformers.ffmpeg_bin_path", return_value="ffprobe")
@patch("ffsubsync.speech_transformers.ffmpeg.probe")
def test_dvd_timings_none_on_failure_or_nothing(mock_probe, mock_bin):
    mock_probe.side_effect = Exception("boom")
    assert _get_dvd_timings_via_ffprobe("test.mkv", "0:s:0") is None
    mock_probe.side_effect = None
    mock_probe.return_value = {"frames": []}
    assert _get_dvd_timings_via_ffprobe("test.mkv", "0:s:0") is None
