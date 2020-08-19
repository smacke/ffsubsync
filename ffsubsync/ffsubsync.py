#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
import logging
import os
import shutil
import subprocess
import sys

import numpy as np

from .aligners import FFTAligner, MaxScoreAligner, FailedToFindAlignmentException
from .constants import *
from .ffmpeg_utils import ffmpeg_bin_path
from .sklearn_shim import Pipeline
from .speech_transformers import (
    VideoSpeechTransformer,
    DeserializeSpeechTransformer,
    make_subtitle_speech_pipeline
)
from .subtitle_parser import make_subtitle_parser
from .subtitle_transformers import SubtitleMerger, SubtitleShifter
from .version import get_version

logger = logging.getLogger(__name__)


def override(args, **kwargs):
    args_dict = dict(args.__dict__)
    args_dict.update(kwargs)
    return args_dict


def _ref_format(ref_fname):
    return ref_fname[-3:]


def make_test_case(args, npy_savename, sync_was_successful):
    if npy_savename is None:
        raise ValueError('need non-null npy_savename')
    tar_dir = '{}.{}'.format(
        args.reference,
        datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    )
    logger.info('creating test archive {}.tar.gz...'.format(tar_dir))
    os.mkdir(tar_dir)
    try:
        shutil.move('ffsubsync.log', tar_dir)
        shutil.copy(args.srtin, tar_dir)
        if sync_was_successful:
            shutil.move(args.srtout, tar_dir)
        if _ref_format(args.reference) in SUBTITLE_EXTENSIONS:
            shutil.copy(args.reference, tar_dir)
        elif args.serialize_speech or args.reference == npy_savename:
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
                return 1
        logger.info('...done')
    finally:
        shutil.rmtree(tar_dir)
    return 0


def try_sync(args, reference_pipe, srt_pipes, result):
    sync_was_successful = True
    try:
        logger.info('extracting speech segments from subtitles file %s...', args.srtin)
        for srt_pipe in srt_pipes:
            srt_pipe.fit(args.srtin)
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
        logger.info('writing output to {}'.format(args.srtout or 'stdout'))
        out_subs.write_file(args.srtout)
    except FailedToFindAlignmentException as e:
        sync_was_successful = False
        logger.error(e)
    else:
        result['offset_seconds'] = offset_seconds
        result['framerate_scale_factor'] = scale_step.scale_factor
    finally:
        result['sync_was_successful'] = sync_was_successful
        return sync_was_successful


def make_reference_pipe(args):
    ref_format = _ref_format(args.reference)
    if ref_format in SUBTITLE_EXTENSIONS:
        if args.vad is not None:
            logger.warning('Vad specified, but reference was not a movie')
        return make_subtitle_speech_pipeline(
            fmt=ref_format,
            **override(
                args,
                encoding=args.reference_encoding or DEFAULT_ENCODING
            )
        )
    elif ref_format in ('npy', 'npz'):
        if args.vad is not None:
            logger.warning('Vad specified, but reference was not a movie')
        return Pipeline([
            ('deserialize', DeserializeSpeechTransformer())
        ])
    else:
        vad = args.vad or DEFAULT_VAD
        if args.reference_encoding is not None:
            logger.warning('Reference srt encoding specified, but reference was a video file')
        ref_stream = args.reference_stream
        if ref_stream is not None and not ref_stream.startswith('0:'):
            ref_stream = '0:' + ref_stream
        return Pipeline([
            ('speech_extract', VideoSpeechTransformer(vad=vad,
                                                      sample_rate=SAMPLE_RATE,
                                                      frame_rate=args.frame_rate,
                                                      start_seconds=args.start_seconds,
                                                      ffmpeg_path=args.ffmpeg_path,
                                                      ref_stream=ref_stream,
                                                      vlc_mode=args.vlc_mode,
                                                      gui_mode=args.gui_mode))
        ])


def make_srt_pipes(args):
    if args.no_fix_framerate:
        framerate_ratios = [1.]
    else:
        framerate_ratios = np.concatenate([
            [1.], np.array(FRAMERATE_RATIOS), 1./np.array(FRAMERATE_RATIOS)
        ])
    parser = make_subtitle_parser(fmt=os.path.splitext(args.srtin)[-1][1:], caching=True, **args.__dict__)
    srt_pipes = [
        make_subtitle_speech_pipeline(
            **override(args, scale_factor=scale_factor, parser=parser)
        )
        for scale_factor in framerate_ratios
    ]
    return srt_pipes


def extract_subtitles_from_reference(args):
    stream = args.extract_subs_from_stream
    if not stream.startswith('0:s:'):
        stream = '0:s:{}'.format(stream)
    elif not stream.startswith('0:') and stream.startswith('s:'):
        stream = '0:{}'.format(stream)
    if not stream.startswith('0:s:'):
        logger.error('invalid stream for subtitle extraction: %s', args.extract_subs_from_stream)
    ffmpeg_args = [ffmpeg_bin_path('ffmpeg', args.gui_mode, ffmpeg_resources_path=args.ffmpeg_path)]
    ffmpeg_args.extend([
        '-y',
        '-nostdin',
        '-loglevel', 'fatal',
        '-i', args.reference,
        '-map', '{}'.format(stream),
        '-f', 'srt',
    ])
    if args.srtout is None:
        ffmpeg_args.append('-')
    else:
        ffmpeg_args.append(args.srtout)
    logger.info('attempting to extract subtitles to {} ...'.format('stdout' if args.srtout is None else args.srtout))
    retcode = subprocess.call(ffmpeg_args)
    if retcode == 0:
        logger.info('...done')
    else:
        logger.error('ffmpeg unable to extract subtitles from reference; return code %d', retcode)
    return retcode


def validate_args(args):
    if args.vlc_mode:
        logger.setLevel(logging.CRITICAL)
    if args.make_test_case and not args.gui_mode:  # this validation not necessary for gui mode
        if args.srtin is None or args.srtout is None:
            raise ValueError('need to specify input and output srt files for test cases')
    if args.overwrite_input:
        if args.extract_subs_from_stream is not None:
            raise ValueError('input overwriting not allowed for extracting subtitles from reference')
        if args.srtin is None:
            raise ValueError(
                'need to specify input srt if --overwrite-input is specified since we cannot overwrite stdin'
            )
        if args.srtout is not None:
            raise ValueError(
                'overwrite input set but output file specified; refusing to run in case this was not intended'
            )
    if args.extract_subs_from_stream is not None:
        if args.make_test_case:
            raise ValueError('test case is for sync and not subtitle extraction')
        if args.srtin is not None:
            raise ValueError('stream specified for reference subtitle extraction; -i flag for sync input not allowed')


def validate_file_permissions(args):
    if not os.access(args.reference, os.R_OK):
        raise ValueError('unable to read reference %s (try checking permissions)' % args.reference)
    if not os.access(args.srtin, os.R_OK):
        raise ValueError('unable to read input subtitles %s (try checking permissions)' % args.srtin)
    if os.path.exists(args.srtout) and not os.access(args.srtout, os.W_OK):
        raise ValueError('unable to write output subtitles %s (try checking permissions)' % args.srtout)
    if args.make_test_case or args.serialize_speech:
        npy_savename = os.path.splitext(args.reference)[0] + '.npz'
        if os.path.exists(npy_savename) and not os.access(npy_savename, os.W_OK):
            raise ValueError('unable to write test case file archive %s (try checking permissions)' % npy_savename)


def run(args):
    result = {
        'retval': 0,
        'offset_seconds': None,
        'framerate_scale_factor': None,
        'sync_was_successful': None
    }
    try:
        validate_args(args)
    except ValueError as e:
        logger.error(e)
        result['retval'] = 1
        return result
    if args.overwrite_input:
        args.srtout = args.srtin
    if args.gui_mode and args.srtout is None:
        args.srtout = '{}.synced.srt'.format(os.path.splitext(args.srtin)[0])
    try:
        validate_file_permissions(args)
    except ValueError as e:
        logger.error(e)
        result['retval'] = 1
        return result
    ref_format = _ref_format(args.reference)
    if args.merge_with_reference and ref_format not in SUBTITLE_EXTENSIONS:
        logger.error('merging synced output with reference only valid '
                     'when reference composed of subtitles')
        result['retval'] = 1
        return result
    if args.make_test_case:
        handler = logging.FileHandler('ffsubsync.log')
        logger.addHandler(handler)
    if args.extract_subs_from_stream is not None:
        result['retval'] = extract_subtitles_from_reference(args)
        return result
    reference_pipe = make_reference_pipe(args)
    logger.info("extracting speech segments from reference '%s'...", args.reference)
    reference_pipe.fit(args.reference)
    logger.info('...done')
    npy_savename = None
    if args.make_test_case or args.serialize_speech:
        logger.info('serializing speech...')
        npy_savename = os.path.splitext(args.reference)[0] + '.npz'
        np.savez_compressed(npy_savename, speech=reference_pipe.transform(args.reference))
        logger.info('...done')
        if args.srtin is None:
            logger.info('unsynchronized subtitle file not specified; skipping synchronization')
            return result
    srt_pipes = make_srt_pipes(args)
    sync_was_successful = try_sync(args, reference_pipe, srt_pipes, result)
    if args.make_test_case:
        handler.close()
        logger.removeHandler(handler)
        result['retval'] += make_test_case(args, npy_savename, sync_was_successful)
    return result


def add_main_args_for_cli(parser):
    parser.add_argument(
        'reference',
        help='Reference (video, subtitles, or a numpy array with VAD speech) to which to synchronize input subtitles.'
    )
    parser.add_argument('-i', '--srtin', help='Input subtitles file (default=stdin).')
    parser.add_argument('-o', '--srtout', help='Output subtitles file (default=stdout).')
    parser.add_argument('--merge-with-reference', '--merge', action='store_true',
                        help='Merge reference subtitles with synced output subtitles.')
    parser.add_argument('--make-test-case', '--create-test-case', action='store_true',
                        help='If specified, serialize reference speech to a numpy array, '
                             'and create an archive with input/output subtitles '
                             'and serialized speech.')
    parser.add_argument(
        '--reference-stream', '--refstream', '--reference-track', '--reftrack',
        default=None,
        help='Which stream/track in the video file to use as reference, '
             'formatted according to ffmpeg conventions. For example, s:0 '
             'uses the first subtitle track; a:3 would use the third audio track.'
    )


def add_cli_only_args(parser):
    parser.add_argument('-v', '--version', action='version',
                        version='{package} {version}'.format(package=__package__, version=get_version()))
    parser.add_argument('--overwrite-input', action='store_true',
                        help='If specified, will overwrite the input srt instead of writing the output to a new file.')
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
    parser.add_argument('--vad', choices=['subs_then_webrtc', 'webrtc', 'subs_then_auditok', 'auditok'],
                        default=None,
                        help='Which voice activity detector to use for speech extraction '
                             '(if using video / audio as a reference, default={}).'.format(DEFAULT_VAD))
    parser.add_argument('--no-fix-framerate', action='store_true',
                        help='If specified, subsync will not attempt to correct a framerate '
                             'mismatch between reference and subtitles.')
    parser.add_argument('--serialize-speech', action='store_true',
                        help='If specified, serialize reference speech to a numpy array.')
    parser.add_argument('--extract-subs-from-stream', default=None,
                        help='If specified, do not attempt sync; instead, just extract subtitles'
                             ' from the specified stream using the reference.')
    parser.add_argument(
        '--ffmpeg-path', '--ffmpegpath', default=None,
        help='Where to look for ffmpeg and ffprobe. Uses the system PATH by default.'
    )
    parser.add_argument('--vlc-mode', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--gui-mode', action='store_true', help=argparse.SUPPRESS)


def make_parser():
    parser = argparse.ArgumentParser(description='Synchronize subtitles with video.')
    add_main_args_for_cli(parser)
    add_cli_only_args(parser)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    return run(args)['retval']


if __name__ == "__main__":
    sys.exit(main())
