#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import sys

import numpy as np
from sklearn.pipeline import Pipeline

from .aligners import FFTAligner, MaxScoreAligner
from .speech_transformers import (
        SubtitleSpeechTransformer,
        VideoSpeechTransformer,
        DeserializeSpeechTransformer
)
from .subtitle_parsers import GenericSubtitleParser, SubtitleOffseter, SubtitleScaler
from .version import __version__

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRAME_RATE = 48000
SAMPLE_RATE = 100

FRAMERATE_RATIOS = [24./23.976, 25./23.976, 25./24.]


def make_srt_parser(args, fmt, encoding=None):
    return GenericSubtitleParser(
        fmt=fmt,
        encoding=encoding or args.encoding,
        max_subtitle_seconds=args.max_subtitle_seconds,
        start_seconds=args.start_seconds
    )


def make_srt_speech_pipeline(args, fmt='srt', scale_factor=1., parser=None):
    if parser is None:
        parser = make_srt_parser(args, fmt)
    assert parser.start_seconds == args.start_seconds
    return Pipeline([
        ('parse', parser),
        ('scale', SubtitleScaler(scale_factor)),
        ('speech_extract', SubtitleSpeechTransformer(
            sample_rate=SAMPLE_RATE,
            start_seconds=args.start_seconds
        ))
    ])


def main():
    parser = argparse.ArgumentParser(description='Synchronize subtitles with video.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))
    parser.add_argument('reference',
                        help='Correct reference (video or srt) to which to sync input subtitles.')
    parser.add_argument('-i', '--srtin', help='Input subtitles file (default=stdin).')
    parser.add_argument('-o', '--srtout', help='Output subtitles file (default=stdout).')
    parser.add_argument('--encoding', default='infer',
                        help='What encoding to use for reading input subtitles.')
    parser.add_argument('--max-subtitle-seconds', type=float, default=10,
                        help='Maximum duration for a subtitle to appear on-screen.')
    parser.add_argument('--start-seconds', type=int, default=0,
                        help='Start time for processing.')
    parser.add_argument('--output-encoding', default='same',
                        help='What encoding to use for writing output subtitles '
                             '(default=same as for input).')
    parser.add_argument('--reference-encoding',
                        help='What encoding to use for reading / writing reference subtitles '
                             '(if applicable).')
    parser.add_argument('--serialize-speech', action='store_true',
                        help='Whether to serialize reference speech to a numpy array.')
    parser.add_argument('--vlc-mode', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.vlc_mode:
        logger.setLevel(logging.CRITICAL)
    if args.reference[-3:] in ('srt', 'ssa', 'ass'):
        fmt = args.reference[-3:]
        reference_pipe = make_srt_speech_pipeline(
            args, parser=make_srt_parser(args, fmt, encoding=args.reference_encoding or 'infer')
        )
    elif args.reference.endswith('npy'):
        reference_pipe = Pipeline([
            ('deserialize', DeserializeSpeechTransformer())
        ])
    else:
        if args.reference_encoding is not None:
            logger.warning('Reference srt encoding specified, but reference was a video file')
        reference_pipe = Pipeline([
            ('speech_extract', VideoSpeechTransformer(sample_rate=SAMPLE_RATE,
                                                      frame_rate=FRAME_RATE,
                                                      start_seconds=args.start_seconds,
                                                      vlc_mode=args.vlc_mode))
        ])
    framerate_ratios = np.concatenate([
        [1.], np.array(FRAMERATE_RATIOS), 1./np.array(FRAMERATE_RATIOS)
    ])
    parser = make_srt_parser(args, fmt=args.srtin[-3:])
    logger.info("extracting speech segments from subtitles '%s'...", args.srtin)
    srt_pipes = [
        make_srt_speech_pipeline(
            args, scale_factor, parser=parser
        ).fit(args.srtin)
        for scale_factor in framerate_ratios
    ]
    logger.info('...done')
    logger.info("extracting speech segments from reference '%s'...", args.reference)
    reference_pipe.fit(args.reference)
    logger.info('...done')
    if args.serialize_speech:
        logger.info('serializing speech...')
        # TODO: better way to substitute extension
        np.save('.'.join(args.reference.split('.')[:-1]) + '.npy',
                reference_pipe.transform(None))
        logger.info('...done')
    logger.info('computing alignments...')
    offset_samples, best_srt_pipe = MaxScoreAligner(FFTAligner).fit_transform(
        reference_pipe.transform(args.reference),
        srt_pipes,
    )
    logger.info('...done')
    offset_seconds = offset_samples / float(SAMPLE_RATE)
    scale_step = best_srt_pipe.named_steps['scale']
    logger.info('offset seconds: %.3f', offset_seconds)
    logger.info('framerate scale factor: %.3f', scale_step.scale_factor)
    offseter = SubtitleOffseter(offset_seconds).fit_transform(scale_step.subs_)
    if args.output_encoding != 'same':
        offseter = offseter.set_encoding(args.output_encoding)
    offseter.write_file(args.srtout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
