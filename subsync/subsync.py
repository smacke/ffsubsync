#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
import logging
import os
import shutil
import sys

import numpy as np
from sklearn.pipeline import Pipeline

from .aligners import FFTAligner, MaxScoreAligner
from .constants import *
from .speech_transformers import (
    VideoSpeechTransformer,
    DeserializeSpeechTransformer,
    make_subtitle_speech_pipeline
)
from .subtitle_parser import make_subtitle_parser
from .subtitle_transformers import SubtitleMerger, SubtitleShifter
from .version import __version__

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def override(args, **kwargs):
    args_dict = dict(args.__dict__)
    args_dict.update(kwargs)
    return args_dict


def run(args):
    retval = 0
    if args.vlc_mode:
        logger.setLevel(logging.CRITICAL)
    if args.make_test_case:
        if args.srtin is None or args.srtout is None:
            logger.error('need to specify input and output srt files for test cases')
            return 1
    ref_format = args.reference[-3:]
    if args.merge_with_reference and ref_format not in SUBTITLE_EXTENSIONS:
        logger.error('merging synced output with reference only valid '
                     'when reference composed of subtitles')
        return 1
    if ref_format in SUBTITLE_EXTENSIONS:
        if args.vad is not None:
            logger.warning('Vad specified, but reference was not a movie')
        reference_pipe = make_subtitle_speech_pipeline(
            fmt=ref_format,
            **override(
                args,
                encoding=args.reference_encoding or DEFAULT_ENCODING
            )
        )
    elif ref_format in ('npy', 'npz'):
        if args.vad is not None:
            logger.warning('Vad specified, but reference was not a movie')
        reference_pipe = Pipeline([
            ('deserialize', DeserializeSpeechTransformer())
        ])
    else:
        vad = args.vad or DEFAULT_VAD
        if args.reference_encoding is not None:
            logger.warning('Reference srt encoding specified, but reference was a video file')
        reference_pipe = Pipeline([
            ('speech_extract', VideoSpeechTransformer(vad=vad,
                                                      sample_rate=SAMPLE_RATE,
                                                      frame_rate=args.frame_rate,
                                                      start_seconds=args.start_seconds,
                                                      vlc_mode=args.vlc_mode))
        ])
    if args.no_fix_framerate:
        framerate_ratios = [1.]
    else:
        framerate_ratios = np.concatenate([
            [1.], np.array(FRAMERATE_RATIOS), 1./np.array(FRAMERATE_RATIOS)
        ])
    logger.info("extracting speech segments from reference '%s'...", args.reference)
    reference_pipe.fit(args.reference)
    logger.info('...done')
    npy_savename = None
    if args.serialize_speech or args.make_test_case:
        logger.info('serializing speech...')
        npy_savename = os.path.splitext(args.reference)[0] + '.npz'
        np.savez_compressed(npy_savename, speech=reference_pipe.transform(args.reference))
        logger.info('...done')
        if args.srtin is None:
            logger.info('unsynchronized subtitle file not specified; skipping synchronization')
            return retval
    parser = make_subtitle_parser(fmt=args.srtin[-3:], caching=True, **args.__dict__)
    logger.info("extracting speech segments from subtitles '%s'...", args.srtin)
    srt_pipes = [
        make_subtitle_speech_pipeline(
            **override(args, scale_factor=scale_factor, parser=parser)
        ).fit(args.srtin)
        for scale_factor in framerate_ratios
    ]
    logger.info('...done')
    logger.info('computing alignments...')
    offset_samples, best_srt_pipe = MaxScoreAligner(
        FFTAligner, SAMPLE_RATE, args.max_offset_seconds
    ).fit_transform(
        reference_pipe.transform(args.reference),
        srt_pipes,
    )
    logger.info('...done')
    offset_seconds = offset_samples / float(SAMPLE_RATE)
    scale_step = best_srt_pipe.named_steps['scale']
    logger.info('offset seconds: %.3f', offset_seconds)
    logger.info('framerate scale factor: %.3f', scale_step.scale_factor)
    output_steps = [('shift', SubtitleShifter(offset_seconds))]
    if args.merge_with_reference:
        output_steps.append(
            ('merge',
             SubtitleMerger(reference_pipe.named_steps['parse'].subs_))
        )
    output_pipe = Pipeline(output_steps)
    out_subs = output_pipe.fit_transform(scale_step.subs_)
    if args.output_encoding != 'same':
        out_subs = out_subs.set_encoding(args.output_encoding)
    out_subs.write_file(args.srtout)
    if args.make_test_case:
        if npy_savename is None:
            raise ValueError('need non-null npy_savename')
        tar_dir = '{}.{}'.format(
            args.reference,
            datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        )
        logger.info('creating test archive {}.tar.gz...'.format(tar_dir))
        os.mkdir(tar_dir)
        try:
            shutil.copy(args.srtin, tar_dir)
            shutil.move(args.srtout, tar_dir)
            if args.serialize_speech or args.reference == npy_savename:
                shutil.copy(npy_savename, tar_dir)
            else:
                shutil.move(npy_savename, tar_dir)
            supported_formats = set(list(zip(*shutil.get_archive_formats()))[0])
            preferred_formats = ['gztar', 'bztar', 'xztar', 'zip', 'tar']
            for archive_format in preferred_formats:
                if archive_format in supported_formats:
                    shutil.make_archive(tar_dir, 'gztar', os.curdir, tar_dir)
                    break
            else:
                logger.error('failed to create test archive; no formats supported '
                             '(this should not happen)')
                retval = 1
            logger.info('...done')
        finally:
            shutil.rmtree(tar_dir)
    return retval


def make_parser():
    parser = argparse.ArgumentParser(description='Synchronize subtitles with video.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))
    parser.add_argument('reference',
                        help='Reference (video, srt, or a numpy array with VAD speech) '
                             'to which to synchronize input subtitles.')
    parser.add_argument('-i', '--srtin', help='Input subtitles file (default=stdin).')
    parser.add_argument('-o', '--srtout', help='Output subtitles file (default=stdout).')
    parser.add_argument('--merge-with-reference', '--merge', action='store_true',
                        help='Merge reference subtitles with synced output subtitles.')
    parser.add_argument('--encoding', default=DEFAULT_ENCODING,
                        help='What encoding to use for reading input subtitles '
                             '(default=%s).' % DEFAULT_ENCODING)
    parser.add_argument('--max-subtitle-seconds', type=float, default=DEFAULT_MAX_SUBTITLE_SECONDS,
                        help='Maximum duration for a subtitle to appear on-screen '
                             '(default=%.3f seconds).' % DEFAULT_MAX_SUBTITLE_SECONDS)
    parser.add_argument('--start-seconds', type=int, default=DEFAULT_START_SECONDS,
                        help='Start time for processing '
                             '(default=%d seconds).' % DEFAULT_START_SECONDS)
    parser.add_argument('--max-offset-seconds', type=int, default=DEFAULT_MAX_OFFSET_SECONDS,
                        help='The max allowed offset seconds for any subtitle segment '
                             '(default=%d seconds).' % DEFAULT_MAX_OFFSET_SECONDS)
    parser.add_argument('--frame-rate', type=int, default=DEFAULT_FRAME_RATE,
                        help='Frame rate for audio extraction (default=%d).' % DEFAULT_FRAME_RATE)
    parser.add_argument('--output-encoding', default='utf-8',
                        help='What encoding to use for writing output subtitles '
                             '(default=utf-8). Can indicate "same" to use same '
                             'encoding as that of the input.')
    parser.add_argument('--reference-encoding',
                        help='What encoding to use for reading / writing reference subtitles '
                             '(if applicable, default=infer).')
    parser.add_argument('--vad', choices=['webrtc', 'auditok'], default=None,
                        help='Which voice activity detector to use for speech extraction '
                             '(if using video / audio as a reference, default=webrtc).')
    parser.add_argument('--no-fix-framerate', action='store_true',
                        help='If specified, subsync will not attempt to correct a framerate '
                             'mismatch between reference and subtitles.')
    parser.add_argument('--make-test-case', '--create-test-case', action='store_true',
                        help='If specified, serialize reference speech to a numpy array, '
                             'and create an archive with input/output subtitles '
                             'and serialized speech.')
    parser.add_argument('--serialize-speech', action='store_true',
                        help='If specified, serialize reference speech to a numpy array.')
    parser.add_argument('--vlc-mode', action='store_true', help=argparse.SUPPRESS)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
