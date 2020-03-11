# -*- coding: utf-8 -*-

import filecmp
import os
import shutil
import tempfile

import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
try:
    import yaml
except ImportError:  # pyyaml does not work with py3.4
    pass

from subsync import subsync
from subsync.speech_transformers import SubtitleSpeechTransformer
from subsync.subtitle_parsers import GenericSubtitleParser

INTEGRATION = 'INTEGRATION'
SYNC_TESTS = 'sync_tests'
REF = 'reference'
SYNCED = 'synchronized'
UNSYNCED = 'unsynchronized'
SKIP = 'skip'
FILECMP = 'filecmp'
SHOULD_DETECT_ENCODING = 'should_detect_encoding'


def gen_args():
    def test_path(fname):
        return os.path.join('test-data', fname)
    if INTEGRATION not in os.environ or os.environ[INTEGRATION] == 0:
        return
    with open('test-data/integration-testing-config.yaml', 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    parser = subsync.make_parser()
    for test in config[SYNC_TESTS]:
        if SKIP in test and test[SKIP]:
            continue
        args = parser.parse_args([test_path(test[REF]), '-i', test_path(test[UNSYNCED])])
        args.truth = test_path(test[SYNCED])
        args.filecmp = True
        if FILECMP in test:
            args.filecmp = test[FILECMP]
        args.should_detect_encoding = None
        if SHOULD_DETECT_ENCODING in test:
            args.should_detect_encoding = test[SHOULD_DETECT_ENCODING]
        yield args


def timestamps_roughly_match(f1, f2):
    parser = GenericSubtitleParser()
    extractor = SubtitleSpeechTransformer(sample_rate=subsync.DEFAULT_FRAME_RATE)
    pipe = make_pipeline(parser, extractor)
    f1_bitstring = pipe.fit_transform(f1).astype(bool)
    f2_bitstring = pipe.fit_transform(f2).astype(bool)
    return np.alltrue(f1_bitstring == f2_bitstring)


def detected_encoding(fname):
    parser = GenericSubtitleParser()
    parser.fit(fname)
    return parser.detected_encoding_


@pytest.mark.integration
@pytest.mark.parametrize('args', gen_args())
def test_sync_matches_ground_truth(args):
    # context manager TemporaryDirectory not available on py2
    dirpath = tempfile.mkdtemp()
    try:
        args.srtout = os.path.join(dirpath, 'test.srt')
        assert subsync.run(args) == 0
        if args.filecmp:
            assert filecmp.cmp(args.srtout, args.truth, shallow=False)
        else:
            assert timestamps_roughly_match(args.srtout, args.truth)
        if args.should_detect_encoding is not None:
            assert detected_encoding(args.srtin) == args.should_detect_encoding
    finally:
        shutil.rmtree(dirpath)
