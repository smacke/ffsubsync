#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import argparse
import logging
import sys

import numpy as np
from sklearn.pipeline import Pipeline

from .aligners import FFTAligner, MaxScoreAligner
from .speech_transformers import SubtitleSpeechTransformer, VideoSpeechTransformer
from .subtitle_parsers import GenericSubtitleParser, SubtitleOffseter, SubtitleScaler
from .version import __version__

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRAME_RATE = 48000
SAMPLE_RATE = 100

FPS_RATIOS = [24./23.976, 25./23.976, 25./24.]


def make_srt_speech_pipeline(fmt, encoding, max_subtitle_seconds, start_seconds, scale_factor):
    return Pipeline([
        ('parse', GenericSubtitleParser(fmt=fmt,
                                        encoding=encoding,
                                        max_subtitle_seconds=max_subtitle_seconds,
                                        start_seconds=start_seconds)),
        ('scale', SubtitleScaler(scale_factor)),
        ('speech_extract', SubtitleSpeechTransformer(sample_rate=SAMPLE_RATE,
                                                     start_seconds=start_seconds))
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
    parser.add_argument('--vlc-mode', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.vlc_mode:
        logger.setLevel(logging.CRITICAL)
    if args.reference[-3:] in ('srt', 'ssa', 'ass'):
        fmt = args.reference[-3:]
        reference_pipe = make_srt_speech_pipeline(fmt,
                                                  args.reference_encoding or 'infer',
                                                  args.max_subtitle_seconds,
                                                  args.start_seconds, 1.0)
    else:
        if args.reference_encoding is not None:
            logger.warning('Reference srt encoding specified, but reference was a video file')
        reference_pipe = Pipeline([
            ('speech_extract', VideoSpeechTransformer(sample_rate=SAMPLE_RATE,
                                                      frame_rate=FRAME_RATE,
                                                      start_seconds=args.start_seconds,
                                                      vlc_mode=args.vlc_mode))
        ])
    fps_ratios = np.concatenate([[1.], np.array(FPS_RATIOS), 1./np.array(FPS_RATIOS)])
    srt_pipes = [
        make_srt_speech_pipeline(args.srtin[-3:],
                                 args.encoding,
                                 args.max_subtitle_seconds,
                                 args.start_seconds,
                                 scale_factor).fit(args.srtin)
        for scale_factor in fps_ratios
    ]
    logger.info('computing alignments...')
    offset_samples, best_srt_pipe = MaxScoreAligner(FFTAligner).fit_transform(
        reference_pipe.fit_transform(args.reference),
        srt_pipes,
    )
    offset_seconds = offset_samples / float(SAMPLE_RATE)
    scale_step = best_srt_pipe.named_steps['scale']
    logger.info('offset seconds: %.3f', offset_seconds)
    logger.info('fps scale factor: %.3f', scale_step.scale_factor)
    offseter = SubtitleOffseter(offset_seconds).fit_transform(scale_step.subs_)
    if args.output_encoding != 'same':
        offseter = offseter.set_encoding(args.output_encoding)
    offseter.write_file(args.srtout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
