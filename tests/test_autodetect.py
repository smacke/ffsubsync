# -*- coding: utf-8 -*-
import argparse
import os

import pytest

from ffsubsync.ffsubsync import (
    _detect_srtin_from_reference,
    _resolve_srtout,
    _run_multiple_references,
    main,
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


def test_parser_accepts_multiple_references():
    args = make_parser().parse_args(
        ["--overwrite-input", "first.mkv", "second.mkv", "third.mkv"]
    )
    assert args.reference == "first.mkv"
    assert args.additional_references == ["second.mkv", "third.mkv"]
    assert args.overwrite_input is True


def test_main_dispatches_multiple_references(monkeypatch):
    monkeypatch.setattr("sys.argv", ["ffs", "first.mkv", "second.mkv"])
    monkeypatch.setattr("ffsubsync.ffsubsync._run_multiple_references", lambda args: 7)
    assert main() == 7


def test_run_multiple_references_processes_each_reference(monkeypatch):
    args = make_parser().parse_args(
        ["--overwrite-input", "first.mkv", "second.mkv"]
    )
    monkeypatch.setattr(
        "ffsubsync.ffsubsync._detect_srtin_from_reference",
        lambda _reference: ["in.srt"],
    )
    seen = []

    def fake_run(reference_args):
        seen.append(
            (
                reference_args.reference,
                reference_args.additional_references,
                reference_args.batch_mode,
            )
        )
        return {"retval": 0}

    monkeypatch.setattr("ffsubsync.ffsubsync.run", fake_run)
    retval = _run_multiple_references(args)

    assert seen == [
        ("first.mkv", [], True),
        ("second.mkv", [], True),
    ]
    assert retval == 0


def test_run_multiple_references_aggregates_failures(monkeypatch):
    args = make_parser().parse_args(["first.mkv", "second.mkv"])
    monkeypatch.setattr(
        "ffsubsync.ffsubsync._detect_srtin_from_reference",
        lambda _reference: ["in.srt"],
    )

    def fake_run(reference_args):
        return {"retval": 1 if reference_args.reference == "second.mkv" else 0}

    monkeypatch.setattr("ffsubsync.ffsubsync.run", fake_run)
    assert _run_multiple_references(args) == 1


def test_run_multiple_references_continues_after_exception(monkeypatch):
    args = make_parser().parse_args(["first.mkv", "second.mkv"])
    monkeypatch.setattr(
        "ffsubsync.ffsubsync._detect_srtin_from_reference",
        lambda _reference: ["in.srt"],
    )
    seen = []

    def fake_run(reference_args):
        seen.append(reference_args.reference)
        if reference_args.reference == "first.mkv":
            raise RuntimeError("broken reference")
        return {"retval": 0}

    monkeypatch.setattr("ffsubsync.ffsubsync.run", fake_run)
    assert _run_multiple_references(args) == 1
    assert seen == ["first.mkv", "second.mkv"]


def test_run_multiple_references_skips_references_without_subtitles(monkeypatch):
    args = make_parser().parse_args(["first.mkv", "second.mkv"])
    monkeypatch.setattr(
        "ffsubsync.ffsubsync._detect_srtin_from_reference",
        lambda reference: [] if reference == "first.mkv" else ["second.srt"],
    )
    seen = []

    def fake_run(reference_args):
        seen.append(reference_args.reference)
        return {"retval": 0}

    monkeypatch.setattr("ffsubsync.ffsubsync.run", fake_run)
    assert _run_multiple_references(args) == 0
    assert seen == ["second.mkv"]


def test_run_multiple_references_rejects_shared_input():
    args = make_parser().parse_args(
        ["first.mkv", "second.mkv", "-i", "shared.srt"]
    )
    assert _run_multiple_references(args) == 1


def test_validate_args_batch_mode_uses_per_reference_inputs(media_dir, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    args = _args(
        str(media_dir / "movie.mkv"), overwrite_input=True, batch_mode=True
    )
    validate_args(args)
    assert {os.path.basename(path) for path in args.srtin} == {
        "movie.srt",
        "movie.en.srt",
    }


def test_validate_args_batch_mode_rejects_missing_input(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    reference = tmp_path / "missing-subs.mkv"
    _touch(str(reference))
    args = _args(str(reference), overwrite_input=True, batch_mode=True)
    with pytest.raises(ValueError, match="no input subtitle files detected"):
        validate_args(args)


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
