import os
from datetime import timedelta
import subprocess
import sys

import pytest
import srt

from ffsubsync.generic_subtitles import GenericSubtitle, GenericSubtitlesFile
from ffsubsync import generic_subtitles


def subtitle_file(content="Hello", encoding="utf-8"):
    sub = srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), content)
    return GenericSubtitlesFile(
        [GenericSubtitle.wrap_inner_subtitle(sub)], sub_format="srt", encoding=encoding
    )


def test_failed_encoding_preserves_original(tmp_path):
    output = tmp_path / "movie.srt"
    output.write_bytes(b"original subtitles")

    with pytest.raises(UnicodeEncodeError):
        subtitle_file("Bonjour à tous", encoding="ascii").write_file(str(output))

    assert output.read_bytes() == b"original subtitles"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux RLIMIT_FSIZE")
def test_failed_write_preserves_original(tmp_path):
    output = tmp_path / "movie.srt"
    output.write_bytes(b"original subtitles")
    script = """
import resource
import signal
import sys
from datetime import timedelta
import srt
from ffsubsync.generic_subtitles import GenericSubtitle, GenericSubtitlesFile

sub = srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Hello")
subs = GenericSubtitlesFile(
    [GenericSubtitle.wrap_inner_subtitle(sub)], sub_format="srt", encoding="utf-8"
)
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
try:
    subs.write_file(sys.argv[1])
except OSError:
    pass
else:
    raise AssertionError("write should fail")
"""
    subprocess.run([sys.executable, "-c", script, str(output)], check=True, timeout=10)

    assert output.read_bytes() == b"original subtitles"
    assert list(tmp_path.iterdir()) == [output]


def test_successful_write_preserves_permissions(tmp_path):
    output = tmp_path / "movie.srt"
    output.write_bytes(b"original subtitles")
    output.chmod(0o640)
    subtitle_file().write_file(str(output))

    assert "Hello" in output.read_text()
    assert output.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.iterdir()) == [output]


def test_failed_replace_preserves_original_and_cleans_temporary_file(
    tmp_path, monkeypatch
):
    output = tmp_path / "movie.srt"
    output.write_bytes(b"original subtitles")

    def fail_replace(source, destination):
        raise PermissionError("cannot replace output")

    monkeypatch.setattr(generic_subtitles.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="cannot replace output"):
        subtitle_file().write_file(str(output))

    assert output.read_bytes() == b"original subtitles"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.skipif(os.name == "nt", reason="symlink creation requires privileges")
def test_write_through_symlink_preserves_link(tmp_path):
    output = tmp_path / "movie.srt"
    output.write_bytes(b"original subtitles")
    link = tmp_path / "link.srt"
    link.symlink_to(output.name)

    subtitle_file().write_file(str(link))

    assert link.is_symlink()
    assert link.read_bytes() == output.read_bytes()
    assert "Hello" in output.read_text()


def test_new_output_file(tmp_path):
    output = tmp_path / "new.srt"
    subtitle_file().write_file(str(output))
    assert "Hello" in output.read_text()
