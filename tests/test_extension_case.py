# -*- coding: utf-8 -*-
import pytest

from ffsubsync import ffsubsync
from ffsubsync.subtitle_parser import GenericSubtitleParser


def _timestamp(seconds):
    return "00:{:02d}:{:02d},000".format(*divmod(seconds, 60))


def _write_srt(path, offset_seconds):
    blocks = []
    start = 0
    for i in range(1, 41):
        # irregular gaps so that only one offset lines the two files up
        start += 3 + (7 * i) % 11
        blocks.append(
            "{}\n{} --> {}\nline {}\n".format(
                i,
                _timestamp(start + offset_seconds),
                _timestamp(start + offset_seconds + 2),
                i,
            )
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


@pytest.mark.parametrize(
    "ref_name, in_name, out_name",
    [
        ("ref.srt", "in.srt", "out.srt"),
        ("ref.SRT", "in.srt", "out.srt"),
        ("ref.srt", "in.SRT", "out.srt"),
        ("ref.srt", "in.srt", "out.SRT"),
    ],
)
def test_sync_with_uppercase_subtitle_extensions(tmp_path, ref_name, in_name, out_name):
    reference = tmp_path / ref_name
    srtin = tmp_path / in_name
    srtout = tmp_path / out_name
    _write_srt(reference, 0)
    _write_srt(srtin, 3)
    args = ffsubsync.make_parser().parse_args(
        [str(reference), "-i", str(srtin), "-o", str(srtout)]
    )
    assert ffsubsync.run(args)["retval"] == 0
    parser = GenericSubtitleParser()
    parser.fit(str(srtout))
    assert parser.subs_[0].start.total_seconds() == pytest.approx(10.0)
