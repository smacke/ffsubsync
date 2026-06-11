# -*- coding: utf-8 -*-
import os
from io import BytesIO

import pytest

from ffsubsync.constants import is_remote_url
from ffsubsync.file_utils import open_file
from ffsubsync.ffsubsync import make_parser, validate_args, validate_file_permissions
from ffsubsync.subtitle_parser import GenericSubtitleParser

SAMPLE_SRT = b"""1
00:00:01,000 --> 00:00:02,000
Hello, world!

2
00:00:03,000 --> 00:00:04,000
Goodbye, world!
"""


@pytest.mark.parametrize(
    "path",
    [
        "http://example.com/video.mp4",
        "https://example.com/video.mp4",
        "rtmp://example.com/live/stream",
        "rtsp://example.com/live/stream",
        "ftp://example.com/video.mp4",
    ],
)
def test_is_remote_url_recognizes_supported_protocols(path):
    assert is_remote_url(path) is True


@pytest.mark.parametrize(
    "path",
    [
        None,
        "movie.mkv",
        "/abs/path/movie.mkv",
        "relative/movie.mkv",
        "C:\\videos\\movie.mkv",
        "ftps://example.com/video.mp4",  # not in the supported set
        "http.mkv",  # local file that merely starts with "http"
    ],
)
def test_is_remote_url_rejects_non_urls(path):
    assert is_remote_url(path) is False


def _args(reference, **overrides):
    args = make_parser().parse_args([reference])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_validate_file_permissions_allows_remote_url():
    # A remote URL has no local file to stat; ffmpeg reads it directly, so
    # permission validation must not reject it.
    args = _args("https://example.com/video.mp4")
    validate_file_permissions(args)  # should not raise


def test_validate_file_permissions_still_rejects_unreadable_local_reference(tmp_path):
    missing = str(tmp_path / "does_not_exist.mkv")
    args = _args(missing)
    with pytest.raises(ValueError, match="read reference"):
        validate_file_permissions(args)


def test_validate_file_permissions_allows_existing_local_reference(tmp_path):
    existing = tmp_path / "movie.mkv"
    existing.write_bytes(b"")
    args = _args(str(existing))
    validate_file_permissions(args)  # should not raise


# guard against the helper accidentally being shadowed by os.path semantics
def test_remote_url_reference_is_not_treated_as_existing_path():
    assert not os.path.exists("https://example.com/video.mp4")


def _patch_urlopen(monkeypatch, payload):
    """Make urllib.request.urlopen return *payload* bytes, recording the request."""
    captured = {}

    def fake_urlopen(req, *args, **kwargs):
        captured["url"] = req.full_url
        captured["headers"] = req.headers
        return BytesIO(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return captured


def test_open_file_routes_remote_url_through_urlopen(monkeypatch):
    url = "https://example.com/subs.srt"
    captured = _patch_urlopen(monkeypatch, SAMPLE_SRT)
    with open_file(url, "rb") as f:
        data = f.read()
    assert data == SAMPLE_SRT
    assert captured["url"] == url
    # a User-Agent is set so picky servers don't reject the request
    assert any(k.lower() == "user-agent" for k in captured["headers"])


def test_open_file_still_reads_local_path(tmp_path):
    p = tmp_path / "subs.srt"
    p.write_bytes(SAMPLE_SRT)
    with open_file(str(p), "rb") as f:
        assert f.read() == SAMPLE_SRT


def test_parser_reads_remote_srt_reference(monkeypatch):
    _patch_urlopen(monkeypatch, SAMPLE_SRT)
    parser = GenericSubtitleParser()
    parser.fit("https://example.com/subs.srt")
    texts = [sub.content for sub in parser.subs_]
    assert texts == ["Hello, world!", "Goodbye, world!"]


def test_validate_args_skips_autodetect_for_remote_url(monkeypatch):
    # No -i given on a tty: local references trigger sibling-srt detection via
    # os.listdir; a remote URL has no listable directory, so detection is skipped
    # rather than crashing.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    args = _args("https://example.com/video.mp4")
    validate_args(args)  # must not raise
    assert not args.srtin
