# -*- coding: utf-8 -*-

import filecmp
import os
import tempfile
import pytest
import yaml
from subsync import subsync


def gen_args():
    def test_path(fname):
        return os.path.join('data', fname)
    with open('data/integration-testing-config.yaml', 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    parser = subsync.make_parser()
    for test in config['sync_tests']:
        args = parser.parse_args([test_path(test['reference']), '-i', test_path(test['unsynchronized'])])
        args.truth = test_path(test['synchronized'])
        yield args


@pytest.mark.end_to_end
@pytest.mark.parametrize('args', gen_args())
def test_sync_matches_ground_truth(args):
    with tempfile.TemporaryDirectory() as dirpath:
        args.srtout = os.path.join(dirpath, 'test.srt')
        assert subsync.run(args) == 0
        assert filecmp.cmp(args.srtout, args.truth, shallow=False)
