# -*- coding: utf-8 -*-
"""
Tests for remote URL support and performance optimization features.
"""
import argparse
import os
import pytest
from unittest.mock import patch, MagicMock

from ffsubsync.constants import (
    REMOTE_URL_PROTOCOLS,
    is_remote_url,
)
from ffsubsync.ffsubsync import (
    validate_remote_url,
    _npy_savename,
)
from ffsubsync.speech_transformers import (
    VideoSpeechTransformer,
)


class TestIsRemoteUrl:
    """Test is_remote_url function."""

    @pytest.mark.parametrize(
        "url, expected",
        [
            # HTTP/HTTPS protocols
            ("http://example.com/video.mp4", True),
            ("https://example.com/video.mp4", True),
            ("https://cdn.example.com/path/to/video.mp4", True),
            # RTMP/RTSP protocols
            ("rtmp://stream.example.com/live/stream", True),
            ("rtsp://camera.example.com/stream1", True),
            # FTP protocol
            ("ftp://files.example.com/video.mp4", True),
            # Local paths (should return False)
            ("/path/to/local/video.mp4", False),
            ("./relative/path/video.mp4", False),
            ("C:\\Windows\\video.mp4", False),
            ("video.mp4", False),
            # None value
            (None, False),
            # Empty string
            ("", False),
        ],
    )
    def test_is_remote_url(self, url, expected):
        assert is_remote_url(url) == expected

    def test_remote_url_protocols_tuple(self):
        """Verify protocol tuple contains expected protocols."""
        assert isinstance(REMOTE_URL_PROTOCOLS, tuple)
        assert "http://" in REMOTE_URL_PROTOCOLS
        assert "https://" in REMOTE_URL_PROTOCOLS
        assert "rtmp://" in REMOTE_URL_PROTOCOLS
        assert "rtsp://" in REMOTE_URL_PROTOCOLS
        assert "ftp://" in REMOTE_URL_PROTOCOLS


class TestValidateRemoteUrl:
    """Test validate_remote_url function."""

    def test_non_http_protocol_returns_true(self):
        """Non-HTTP protocols should return True (skip check)."""
        assert validate_remote_url("rtmp://stream.example.com/live") is True
        assert validate_remote_url("rtsp://camera.example.com/stream") is True

    @patch("urllib.request.urlopen")
    def test_head_request_success(self, mock_urlopen):
        """HEAD request success should return True."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        assert validate_remote_url("https://example.com/video.mp4") is True

    @patch("urllib.request.urlopen")
    def test_url_not_accessible(self, mock_urlopen):
        """Inaccessible URL should return False."""
        mock_urlopen.side_effect = Exception("Connection failed")

        assert validate_remote_url("https://invalid.example.com/video.mp4") is False


class TestNpySavename:
    """Test _npy_savename function."""

    def test_local_path(self):
        """Local path should be processed normally."""
        args = argparse.Namespace(reference="/path/to/video.mp4")
        result = _npy_savename(args)
        assert result == "/path/to/video.npz"

    def test_remote_url_with_filename(self):
        """Remote URL with filename should extract filename."""
        args = argparse.Namespace(reference="https://example.com/path/video.mp4")
        result = _npy_savename(args)
        assert result == "video.npz"

    def test_remote_url_without_filename(self):
        """Remote URL without filename should use domain + timestamp."""
        args = argparse.Namespace(reference="https://example.com/")
        result = _npy_savename(args)
        # Should contain domain (dots replaced with underscores)
        assert result.startswith("example_com_")
        assert result.endswith(".npz")
        # Should contain timestamp format
        assert len(result) > len("example_com_.npz")


class TestVideoSpeechTransformerParams:
    """Test VideoSpeechTransformer new parameters."""

    def test_default_params(self):
        """Test default parameter values."""
        transformer = VideoSpeechTransformer(
            vad="webrtc",
            sample_rate=100,
            frame_rate=48000,
            non_speech_label=0.0,
        )
        assert transformer.extract_audio_first is False
        assert transformer.max_duration_seconds is None
        assert transformer._temp_audio_file is None

    def test_extract_audio_first_param(self):
        """Test extract_audio_first parameter."""
        transformer = VideoSpeechTransformer(
            vad="webrtc",
            sample_rate=100,
            frame_rate=48000,
            non_speech_label=0.0,
            extract_audio_first=True,
        )
        assert transformer.extract_audio_first is True

    def test_max_duration_seconds_param(self):
        """Test max_duration_seconds parameter."""
        transformer = VideoSpeechTransformer(
            vad="webrtc",
            sample_rate=100,
            frame_rate=48000,
            non_speech_label=0.0,
            max_duration_seconds=600,
        )
        assert transformer.max_duration_seconds == 600


class TestSpeechTransformersIsRemoteUrl:
    """Test is_remote_url is properly imported in speech_transformers module."""

    def test_is_remote_url_imported(self):
        """Verify is_remote_url can be used from speech_transformers context."""
        from ffsubsync.speech_transformers import is_remote_url as st_is_remote_url
        assert st_is_remote_url("https://example.com/video.mp4") is True
        assert st_is_remote_url("/local/path.mp4") is False
        assert st_is_remote_url(None) is False


class TestRemoteUrlIntegration:
    """Integration tests for remote URL features."""

    def test_url_detection_and_savename_consistency(self):
        """URL detection and filename generation should be consistent."""
        test_urls = [
            "https://example.com/video.mp4",
            "http://cdn.example.com/path/to/movie.mkv",
            "https://storage.example.com/",
        ]
        
        for url in test_urls:
            assert is_remote_url(url) is True
            args = argparse.Namespace(reference=url)
            result = _npy_savename(args)
            assert result.endswith(".npz")
            # Should not contain URL special characters
            assert "://" not in result
            assert "?" not in result
            assert "&" not in result


class TestMultiSegmentSync:
    """Tests for multi-segment sync feature."""

    def test_compute_weighted_median_offset_basic(self):
        """Test weighted median offset computation."""
        from ffsubsync.aligners import compute_weighted_median_offset
        
        segment_results = [
            {'offset_seconds': 2.0, 'score': 0.9, 'segment_start': 0},
            {'offset_seconds': 2.1, 'score': 0.85, 'segment_start': 60},
            {'offset_seconds': 2.0, 'score': 0.88, 'segment_start': 120},
            {'offset_seconds': 2.05, 'score': 0.92, 'segment_start': 180},
        ]
        
        median_offset, combined_score, valid_segments = compute_weighted_median_offset(
            segment_results, min_score=0.3
        )
        
        assert len(valid_segments) == 4
        assert 1.9 <= median_offset <= 2.2
        assert combined_score > 0.8

    def test_compute_weighted_median_offset_filters_low_scores(self):
        """Test that low score segments are filtered."""
        from ffsubsync.aligners import compute_weighted_median_offset
        
        segment_results = [
            {'offset_seconds': 2.0, 'score': 0.9, 'segment_start': 0},
            {'offset_seconds': 5.0, 'score': 0.1, 'segment_start': 60},  # Low score - noise
            {'offset_seconds': 2.1, 'score': 0.85, 'segment_start': 120},
        ]
        
        median_offset, combined_score, valid_segments = compute_weighted_median_offset(
            segment_results, min_score=0.3
        )
        
        # Should only use 2 segments (filtered out low score)
        assert len(valid_segments) == 2
        assert 1.9 <= median_offset <= 2.2

    def test_compute_weighted_median_offset_no_valid_segments(self):
        """Test exception when no valid segments."""
        from ffsubsync.aligners import compute_weighted_median_offset, FailedToFindAlignmentException
        
        segment_results = [
            {'offset_seconds': 2.0, 'score': -1, 'segment_start': 0},
            {'offset_seconds': 2.1, 'score': -0.5, 'segment_start': 60},
        ]
        
        with pytest.raises(FailedToFindAlignmentException):
            compute_weighted_median_offset(segment_results, min_score=0.3)

    def test_video_speech_transformer_multi_segment_params(self):
        """Test VideoSpeechTransformer with multi-segment params."""
        transformer = VideoSpeechTransformer(
            vad="webrtc",
            sample_rate=100,
            frame_rate=48000,
            non_speech_label=0.0,
            multi_segment_sync=True,
            segment_count=10,
        )
        assert transformer.multi_segment_sync is True
        assert transformer.segment_count == 10

    def test_calculate_segment_positions(self):
        """Test segment position calculation without margins (default)."""
        transformer = VideoSpeechTransformer(
            vad="webrtc",
            sample_rate=100,
            frame_rate=48000,
            non_speech_label=0.0,
            segment_count=5,
        )
        
        # Test with long video (1 hour) - default behavior: no margins
        segments = transformer._calculate_segment_positions(3600)
        assert len(segments) == 5
        # First segment should start at 0 (no margin by default)
        assert segments[0]['start'] == 0
        # Segments should be distributed across the video
        assert segments[-1]['start'] > 0

    def test_calculate_segment_positions_with_skip_intro_outro(self):
        """Test segment position calculation with skip_intro_outro enabled."""
        transformer = VideoSpeechTransformer(
            vad="webrtc",
            sample_rate=100,
            frame_rate=48000,
            non_speech_label=0.0,
            segment_count=5,
            skip_intro_outro=True,
        )
        
        # Test with long video (1 hour) - with margins enabled
        segments = transformer._calculate_segment_positions(3600)
        assert len(segments) == 5
        # First segment should start after start margin (30s default)
        assert segments[0]['start'] >= transformer.DEFAULT_START_MARGIN
        # Last segment should end before end margin
        last_end = segments[-1]['start'] + segments[-1]['duration']
        assert last_end <= 3600 - transformer.DEFAULT_END_MARGIN + transformer.SEGMENT_DURATION
        
    def test_calculate_segment_positions_short_video(self):
        """Test segment position calculation for short video."""
        transformer = VideoSpeechTransformer(
            vad="webrtc",
            sample_rate=100,
            frame_rate=48000,
            non_speech_label=0.0,
            segment_count=8,
        )
        
        # Test with short video (3 minutes) - should reduce segment count
        segments = transformer._calculate_segment_positions(180)
        assert len(segments) < 8  # Should be reduced
        assert len(segments) >= 1  # At least 1 segment

    def test_calculate_segment_positions_margins_exceed_duration(self):
        """Test segment calculation when margins exceed video duration."""
        transformer = VideoSpeechTransformer(
            vad="webrtc",
            sample_rate=100,
            frame_rate=48000,
            non_speech_label=0.0,
            segment_count=5,
        )
        
        # Test with very short video (60 seconds) - margins should be reduced
        segments = transformer._calculate_segment_positions(60)
        assert len(segments) >= 1
        # All segments should be within bounds
        for seg in segments:
            assert seg['start'] >= 0
            assert seg['start'] + seg['duration'] <= 60

    def test_calculate_segment_positions_boundary_checks(self):
        """Test boundary checks in segment calculation."""
        transformer = VideoSpeechTransformer(
            vad="webrtc",
            sample_rate=100,
            frame_rate=48000,
            non_speech_label=0.0,
            segment_count=10,
        )
        
        # Test with medium video (10 minutes)
        segments = transformer._calculate_segment_positions(600)
        
        # All segments should be within bounds
        for seg in segments:
            assert seg['start'] >= 0, f"Segment start {seg['start']} is negative"
            assert seg['start'] + seg['duration'] <= 600, \
                f"Segment end {seg['start'] + seg['duration']} exceeds video duration"
        
        # Segments should not overlap significantly
        for i in range(len(segments) - 1):
            # Allow some overlap but not complete overlap
            assert segments[i+1]['start'] >= segments[i]['start'], \
                "Segments should be in order"
