# -*- coding: utf-8 -*-

import filecmp
import os
import shutil
import tempfile

import numpy as np
import pytest
try:
    import yaml
except ImportError:  # pyyaml does not work with py3.4
    pass

from subsync import subsync
from subsync.sklearn_shim import make_pipeline
from subsync.speech_transformers import SubtitleSpeechTransformer
from subsync.subtitle_parser import GenericSubtitleParser

INTEGRATION = 'INTEGRATION'
SYNC_TESTS = 'sync_tests'
REF = 'reference'
SYNCED = 'synchronized'
UNSYNCED = 'unsynchronized'
SKIP = 'skip'
FILECMP = 'filecmp'
SHOULD_DETECT_ENCODING = 'should_detect_encoding'
EXTRA_ARGS = 'extra_args'
EXTRA_NO_VALUE_ARGS = 'extra_no_value_args'


def gen_synctest_configs():
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
        unparsed_args = [test_path(test[REF]), '-i', test_path(test[UNSYNCED])]
        if EXTRA_ARGS in test:
            for extra_key, extra_value in test[EXTRA_ARGS].items():
                unparsed_args.extend(['--{}'.format(extra_key), str(extra_value)])
        if EXTRA_NO_VALUE_ARGS in test:
            for extra_key in test[EXTRA_NO_VALUE_ARGS]:
                unparsed_args.append('--{}'.format(extra_key))
        args = parser.parse_args(unparsed_args)
        truth = test_path(test[SYNCED])
        should_filecmp = True
        if FILECMP in test:
            should_filecmp = test[FILECMP]
        should_detect_encoding = None
        if SHOULD_DETECT_ENCODING in test:
            should_detect_encoding = test[SHOULD_DETECT_ENCODING]
        yield args, truth, should_filecmp, should_detect_encoding


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
@pytest.mark.parametrize('args,truth,should_filecmp,should_detect_encoding', gen_synctest_configs())
def test_sync_matches_ground_truth(args, truth, should_filecmp, should_detect_encoding):
    # context manager TemporaryDirectory not available on py2
    dirpath = tempfile.mkdtemp()
    try:
        args.srtout = os.path.join(dirpath, 'test.srt')
        assert subsync.run(args) == 0
        if should_filecmp:
            assert filecmp.cmp(args.srtout, truth, shallow=False)
        else:
            assert timestamps_roughly_match(args.srtout, truth)
        if should_detect_encoding is not None:
            assert detected_encoding(args.srtin) == should_detect_encoding
    finally:
        shutil.rmtree(dirpath)
