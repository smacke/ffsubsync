# -*- coding: utf-8 -*-
import argparse
import os

import pytest

from ffsubsync.ffsubsync import (
    _detect_srtin_from_reference,
    _resolve_srtout,
    make_parser,
    validate_args,
)


def _touch(*paths):
    for path in paths:
        open(path, "w").close()


@pytest.fixture
def media_dir(tmp_path):
    """A reference video surrounded by a mix of matching/non-matching srt files."""
    d = tmp_path / "media"
    d.mkdir()
    _touch(
        str(d / "movie.mkv"),
        str(d / "movie.srt"),  # match
        str(d / "movie.en.srt"),  # match (language suffix)
        str(d / "movie.synced.srt"),  # excluded: our own prior output
        str(d / "movie_backup.srt"),  # excluded: not a `.`-delimited suffix
        str(d / "other.srt"),  # excluded: unrelated name
    )
    return d


def test_detect_matches_only_name_sharing_srts(media_dir):
    matched = {
        os.path.basename(p)
        for p in _detect_srtin_from_reference(str(media_dir / "movie.mkv"))
    }
    assert matched == {"movie.srt", "movie.en.srt"}


def test_detect_is_independent_of_cwd(media_dir, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    matched = _detect_srtin_from_reference(str(media_dir / "movie.mkv"))
    # results are returned as paths rooted at the reference's own directory
    assert {os.path.basename(p) for p in matched} == {"movie.srt", "movie.en.srt"}
    assert all(os.path.dirname(p) == str(media_dir) for p in matched)


def test_detect_excludes_reference_itself_when_reference_is_srt(media_dir):
    matched = {
        os.path.basename(p)
        for p in _detect_srtin_from_reference(str(media_dir / "movie.srt"))
    }
    assert "movie.srt" not in matched
    assert matched == {"movie.en.srt"}


def test_detect_returns_empty_when_nothing_matches(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    _touch(str(d / "movie.mkv"), str(d / "unrelated.srt"))
    assert _detect_srtin_from_reference(str(d / "movie.mkv")) == []


def _args(reference, **overrides):
    args = make_parser().parse_args([reference])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_validate_args_auto_detects_and_derives_output(media_dir, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    args = _args(str(media_dir / "movie.mkv"))
    validate_args(args)
    assert {os.path.basename(p) for p in args.srtin} == {"movie.srt", "movie.en.srt"}
    assert args.auto_srtout is True
    assert args.overwrite_input is False  # non-destructive by default
    assert args.srtout is None


def test_validate_args_respects_overwrite_input(media_dir, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    args = _args(str(media_dir / "movie.mkv"), overwrite_input=True)
    validate_args(args)
    assert args.srtin  # detected
    assert getattr(args, "auto_srtout", False) is False


def test_validate_args_multiple_detected_with_output_errors(media_dir, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    args = _args(str(media_dir / "movie.mkv"), srtout="out.srt")
    with pytest.raises(ValueError, match="multiple input srt files"):
        validate_args(args)


def test_validate_args_skips_detection_when_stdin_piped(media_dir, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    args = _args(str(media_dir / "movie.mkv"))
    validate_args(args)
    assert not args.srtin  # left for the stdin code path
    assert getattr(args, "auto_srtout", False) is False


def test_validate_args_no_match_leaves_srtin_unset(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    d = tmp_path / "empty"
    d.mkdir()
    _touch(str(d / "movie.mkv"))
    args = _args(str(d / "movie.mkv"))
    validate_args(args)
    assert not args.srtin
    assert getattr(args, "auto_srtout", False) is False


@pytest.mark.parametrize(
    "overwrite_input, auto_srtout, srtout, srtin, expected",
    [
        (True, False, None, "in.srt", "in.srt"),  # overwrite wins
        (False, True, None, "movie.en.srt", "movie.en.synced.srt"),  # derived
        (False, True, None, None, None),  # stdin input has no derived name
        (False, False, "out.srt", "in.srt", "out.srt"),  # explicit output
        (False, False, None, "in.srt", None),  # default stdout
    ],
)
def test_resolve_srtout(overwrite_input, auto_srtout, srtout, srtin, expected):
    args = argparse.Namespace(
        overwrite_input=overwrite_input, auto_srtout=auto_srtout, srtout=srtout
    )
    assert _resolve_srtout(args, srtin) == expected
