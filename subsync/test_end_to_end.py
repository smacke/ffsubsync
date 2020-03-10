# -*- coding: utf-8 -*-

import filecmp
import os
import tempfile
import pytest
from subsync import subsync


def gen_args():
    parser = subsync.make_parser()
    args = parser.parse_args(['data/28_days_later.npz', '-i', 'data/28_days_later.unsynced.srt'])
    args.truth = 'data/28_days_later.synced.srt'
    yield args


@pytest.mark.parametrize('args', gen_args())
def test_end_to_end(args):
    with tempfile.TemporaryDirectory() as dirpath:
        args.srtout = os.path.join(dirpath, 'test.srt')
        assert subsync.run(args) == 0
        assert filecmp.cmp(args.srtout, args.truth, shallow=False)
