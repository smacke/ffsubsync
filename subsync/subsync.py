#!/usr/bin/env python
from __future__ import print_function
from builtins import range

import argparse
import math
import logging
import sys

import numpy as np
from sklearn.pipeline import Pipeline

from .aligners import FFTAligner, MaxScoreAligner
from .speech_transformers import SubtitleSpeechTransformer, VideoSpeechTransformer
from .subtitle_parsers import SrtParser, SrtOffseter
from .version import __version__

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRAME_RATE = 48000
SAMPLE_RATE = 100


def make_srt_speech_pipeline(encoding='infer'):
    return Pipeline([
        ('parse', SrtParser(encoding=encoding)),
        ('speech_extract', SubtitleSpeechTransformer(sample_rate=SAMPLE_RATE))
    ])


def main():
    parser = argparse.ArgumentParser(description='Synchronize subtitles with video.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))
    parser.add_argument('reference')
    parser.add_argument('-i', '--srtin', required=True)  # TODO: allow read from stdin
    parser.add_argument('-o', '--srtout', default=None)
    parser.add_argument('--vlc-mode', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.vlc_mode:
        logger.setLevel(logging.CRITICAL)
    if args.reference.endswith('srt'):
        reference_pipe = make_srt_speech_pipeline()
    else:
        reference_pipe = Pipeline([
            ('speech_extract', VideoSpeechTransformer(sample_rate=SAMPLE_RATE,
                                                      frame_rate=FRAME_RATE,
                                                      vlc_mode=args.vlc_mode))
        ])
    srtin_pipe = make_srt_speech_pipeline()
    logger.info('computing alignments...')
    offset_seconds = MaxScoreAligner(FFTAligner).fit_transform(
        srtin_pipe.fit_transform(args.srtin),
        reference_pipe.fit_transform(args.reference)
    ) / 100.
    logger.info('offset seconds: %.3f', offset_seconds)
    SrtOffseter(offset_seconds).fit_transform(
        srtin_pipe.named_steps['parse'].subs_).write_file(args.srtout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
