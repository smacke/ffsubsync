# -*- coding: utf-8 -*-

import filecmp
import os
import shutil
import tempfile
import pytest
try:
    import yaml
except ImportError:  # pyyaml does not work with py3.4
    pass
from subsync import subsync

SYNC_TESTS = 'sync_tests'
REF = 'reference'
SYNCED = 'synchronized'
UNSYNCED = 'unsynchronized'


def gen_args():
    def test_path(fname):
        return os.path.join('test-data', fname)
    if 'INTEGRATION' not in os.environ or os.environ['INTEGRATION'] == 0:
        return
    with open('test-data/integration-testing-config.yaml', 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    parser = subsync.make_parser()
    for test in config[SYNC_TESTS]:
        args = parser.parse_args([test_path(test[REF]), '-i', test_path(test[UNSYNCED])])
        args.truth = test_path(test[SYNCED])
        yield args


@pytest.mark.integration
@pytest.mark.parametrize('args', gen_args())
def test_sync_matches_ground_truth(args):
    # context manager TemporaryDirectory not available on py2
    dirpath = tempfile.mkdtemp()
    try:
        args.srtout = os.path.join(dirpath, 'test.srt')
        assert subsync.run(args) == 0
        assert filecmp.cmp(args.srtout, args.truth, shallow=False)
    finally:
        shutil.rmtree(dirpath)
