#!/usr/bin/env python
import argparse
import logging
import sys

from sklearn.pipeline import Pipeline

from .aligners import FFTAligner, MaxScoreAligner
from .speech_transformers import SubtitleSpeechTransformer, VideoSpeechTransformer
from .subtitle_parsers import SrtParser, SrtOffseter
from .version import __version__

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRAME_RATE = 48000
SAMPLE_RATE = 100


def make_srt_speech_pipeline(encoding, max_subtitle_seconds):
    return Pipeline([
        ('parse', SrtParser(encoding=encoding, max_subtitle_seconds=max_subtitle_seconds)),
        ('speech_extract', SubtitleSpeechTransformer(sample_rate=SAMPLE_RATE))
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
    if args.reference.endswith('srt'):
        reference_pipe = make_srt_speech_pipeline(args.reference_encoding or 'infer')
    else:
        if args.reference_encoding is not None:
            logger.warning('Reference srt encoding specified, but reference was a video file')
        reference_pipe = Pipeline([
            ('speech_extract', VideoSpeechTransformer(sample_rate=SAMPLE_RATE,
                                                      frame_rate=FRAME_RATE,
                                                      vlc_mode=args.vlc_mode))
        ])
    srtin_pipe = make_srt_speech_pipeline(args.encoding, args.max_subtitle_seconds)
    logger.info('computing alignments...')
    offset_seconds = MaxScoreAligner(FFTAligner).fit_transform(
        srtin_pipe.fit_transform(args.srtin),
        reference_pipe.fit_transform(args.reference)
    ) / float(SAMPLE_RATE)
    logger.info('offset seconds: %.3f', offset_seconds)
    SrtOffseter(offset_seconds).fit_transform(
        srtin_pipe.named_steps['parse'].subs_).set_encoding(
        args.output_encoding).write_file(args.srtout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
