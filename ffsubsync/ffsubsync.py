#!/usr/bin/env python
# -*- coding: future_annotations -*-
import argparse
from datetime import datetime
import logging
import os
import shutil
import subprocess
import sys
from typing import cast, Any, Callable, Dict, Optional, Union

import numpy as np

from ffsubsync.aligners import FFTAligner, MaxScoreAligner, FailedToFindAlignmentException
from ffsubsync.constants import *
from ffsubsync.ffmpeg_utils import ffmpeg_bin_path
from ffsubsync.sklearn_shim import Pipeline, TransformerMixin
from ffsubsync.speech_transformers import (
    VideoSpeechTransformer,
    DeserializeSpeechTransformer,
    make_subtitle_speech_pipeline
)
from ffsubsync.subtitle_parser import make_subtitle_parser
from ffsubsync.subtitle_transformers import SubtitleMerger, SubtitleShifter
from ffsubsync.version import get_version

logger: logging.Logger = logging.getLogger(__name__)


def override(args: argparse.Namespace, **kwargs: Any) -> Dict[str, Any]:
    args_dict = dict(args.__dict__)
    args_dict.update(kwargs)
    return args_dict


def _ref_format(ref_fname: str) -> str:
    return ref_fname[-3:]


def make_test_case(args: argparse.Namespace, npy_savename: Optional[str], sync_was_successful: bool) -> int:
    if npy_savename is None:
        raise ValueError('need non-null npy_savename')
    tar_dir = '{}.{}'.format(
        args.reference,
        datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    )
    logger.info('creating test archive {}.tar.gz...'.format(tar_dir))
    os.mkdir(tar_dir)
    try:
        log_path = 'ffsubsync.log'
        if args.log_dir_path and os.path.isdir(args.log_dir_path):
            log_path = os.path.join(args.log_dir_path, log_path)
        shutil.copy(log_path, tar_dir)
        shutil.copy(args.srtin[0], tar_dir)
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
                shutil.make_archive(tar_dir, archive_format, os.curdir, tar_dir)
                break
        else:
            logger.error('failed to create test archive; no formats supported '
                         '(this should not happen)')
            return 1
        logger.info('...done')
    finally:
        shutil.rmtree(tar_dir)
    return 0


def get_srt_pipe_maker(
    args: argparse.Namespace, srtin: Optional[str]
) -> Callable[[Optional[float]], Union[Pipeline, Callable[[float], Pipeline]]]:
    if srtin is None:
        srtin_format = 'srt'
    else:
        srtin_format = os.path.splitext(srtin)[-1][1:]
    parser = make_subtitle_parser(fmt=srtin_format, caching=True, **args.__dict__)
    return lambda scale_factor: make_subtitle_speech_pipeline(
        **override(args, scale_factor=scale_factor, parser=parser)
    )


def get_framerate_ratios_to_try(args: argparse.Namespace) -> List[Optional[float]]:
    if args.no_fix_framerate:
        return []
    else:
        framerate_ratios = list(np.concatenate([
            np.array(FRAMERATE_RATIOS), 1./np.array(FRAMERATE_RATIOS)
        ]))
        if args.gss:
            framerate_ratios.append(None)
        return framerate_ratios


def try_sync(args: argparse.Namespace, reference_pipe: Pipeline, result: Dict[str, Any]) -> bool:
    sync_was_successful = True
    exc = None
    try:
        logger.info('extracting speech segments from %s...',
                    'stdin' if not args.srtin else 'subtitles file(s) {}'.format(args.srtin))
        if not args.srtin:
            args.srtin = [None]
        for srtin in args.srtin:
            srtout = srtin if args.overwrite_input else args.srtout
            srt_pipe_maker = get_srt_pipe_maker(args, srtin)
            framerate_ratios = get_framerate_ratios_to_try(args)
            srt_pipes = [srt_pipe_maker(1.)] + [srt_pipe_maker(rat) for rat in framerate_ratios]
            for srt_pipe in srt_pipes:
                if callable(srt_pipe):
                    continue
                else:
                    srt_pipe.fit(srtin)
            if not args.skip_infer_framerate_ratio and hasattr(reference_pipe[-1], 'num_frames'):
                inferred_framerate_ratio_from_length = float(reference_pipe[-1].num_frames) / cast(Pipeline, srt_pipes[0])[-1].num_frames
                logger.info('inferred frameratio ratio: %.3f' % inferred_framerate_ratio_from_length)
                srt_pipes.append(cast(Pipeline, srt_pipe_maker(inferred_framerate_ratio_from_length)).fit(srtin))
                logger.info('...done')
            logger.info('computing alignments...')
            if args.skip_sync:
                best_score = 0.
                best_srt_pipe = cast(Pipeline, srt_pipes[0])
                offset_samples = 0
            else:
                (best_score, offset_samples), best_srt_pipe = MaxScoreAligner(
                    FFTAligner, srtin, SAMPLE_RATE, args.max_offset_seconds
                ).fit_transform(
                    reference_pipe.transform(args.reference),
                    srt_pipes,
                )
            logger.info('...done')
            offset_seconds = offset_samples / float(SAMPLE_RATE) + args.apply_offset_seconds
            scale_step = best_srt_pipe.named_steps['scale']
            logger.info('score: %.3f', best_score)
            logger.info('offset seconds: %.3f', offset_seconds)
            logger.info('framerate scale factor: %.3f', scale_step.scale_factor)
            output_steps: List[Tuple[str, TransformerMixin]] = [('shift', SubtitleShifter(offset_seconds))]
            if args.merge_with_reference:
                output_steps.append(
                    ('merge', SubtitleMerger(reference_pipe.named_steps['parse'].subs_))
                )
            output_pipe = Pipeline(output_steps)
            out_subs = output_pipe.fit_transform(scale_step.subs_)
            if args.output_encoding != 'same':
                out_subs = out_subs.set_encoding(args.output_encoding)
            logger.info('writing output to {}'.format(srtout or 'stdout'))
            out_subs.write_file(srtout)
    except FailedToFindAlignmentException as e:
        sync_was_successful = False
        logger.error(e)
    except Exception as e:
        exc = e
        sync_was_successful = False
        logger.error(e)
    else:
        result['offset_seconds'] = offset_seconds
        result['framerate_scale_factor'] = scale_step.scale_factor
    finally:
        if exc is not None:
            raise exc
        result['sync_was_successful'] = sync_was_successful
        return sync_was_successful


def make_reference_pipe(args: argparse.Namespace) -> Pipeline:
    ref_format = _ref_format(args.reference)
    if ref_format in SUBTITLE_EXTENSIONS:
        if args.vad is not None:
            logger.warning('Vad specified, but reference was not a movie')
        return cast(Pipeline, make_subtitle_speech_pipeline(
            fmt=ref_format,
            **override(
                args,
                encoding=args.reference_encoding or DEFAULT_ENCODING
            )
        ))
    elif ref_format in ('npy', 'npz'):
        if args.vad is not None:
            logger.warning('Vad specified, but reference was not a movie')
        return Pipeline([
            ('deserialize', DeserializeSpeechTransformer(args.non_speech_label))
        ])
    else:
        vad = args.vad or DEFAULT_VAD
        if args.reference_encoding is not None:
            logger.warning('Reference srt encoding specified, but reference was a video file')
        ref_stream = args.reference_stream
        if ref_stream is not None and not ref_stream.startswith('0:'):
            ref_stream = '0:' + ref_stream
        return Pipeline([
            ('speech_extract', VideoSpeechTransformer(
                vad=vad,
                sample_rate=SAMPLE_RATE,
                frame_rate=args.frame_rate,
                non_speech_label=args.non_speech_label,
                start_seconds=args.start_seconds,
                ffmpeg_path=args.ffmpeg_path,
                ref_stream=ref_stream,
                vlc_mode=args.vlc_mode,
                gui_mode=args.gui_mode
            )),
        ])


def extract_subtitles_from_reference(args: argparse.Namespace) -> int:
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


def validate_args(args: argparse.Namespace) -> None:
    if args.vlc_mode:
        logger.setLevel(logging.CRITICAL)
    if args.srtin:
        if len(args.srtin) > 1 and not args.overwrite_input:
                raise ValueError('cannot specify multiple input srt files without overwriting')
        if len(args.srtin) > 1 and args.make_test_case:
                raise ValueError('cannot specify multiple input srt files for test cases')
        if len(args.srtin) > 1 and args.gui_mode:
                raise ValueError('cannot specify multiple input srt files in GUI mode')
    if args.make_test_case and not args.gui_mode:  # this validation not necessary for gui mode
        if args.srtin is None or args.srtout is None:
            raise ValueError('need to specify input and output srt files for test cases')
    if args.overwrite_input:
        if args.extract_subs_from_stream is not None:
            raise ValueError('input overwriting not allowed for extracting subtitles from reference')
        if not args.srtin:
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
        if args.srtin:
            raise ValueError('stream specified for reference subtitle extraction; -i flag for sync input not allowed')


def validate_file_permissions(args: argparse.Namespace) -> None:
    error_string_template = 'unable to {action} {file}; try ensuring file exists and has correct permissions'
    if not os.access(args.reference, os.R_OK):
        raise ValueError(error_string_template.format(action='read reference', file=args.reference))
    if args.srtin:
        for srtin in args.srtin:
            if srtin is not None and not os.access(srtin, os.R_OK):
                raise ValueError(error_string_template.format(action='read input subtitles', file=srtin))
    if args.srtout is not None and os.path.exists(args.srtout) and not os.access(args.srtout, os.W_OK):
        raise ValueError(error_string_template.format(action='write output subtitles', file=args.srtout))
    if args.make_test_case or args.serialize_speech:
        npy_savename = os.path.splitext(args.reference)[0] + '.npz'
        if os.path.exists(npy_savename) and not os.access(npy_savename, os.W_OK):
            raise ValueError('unable to write test case file archive %s (try checking permissions)' % npy_savename)


def run(args: argparse.Namespace) -> Dict[str, Any]:
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
    if args.gui_mode and args.srtout is None:
        args.srtout = '{}.synced.srt'.format(os.path.splitext(args.srtin[0])[0])
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
    log_handler = None
    log_path = None
    if args.make_test_case:
        log_path = 'ffsubsync.log'
        if args.log_dir_path and os.path.isdir(args.log_dir_path):
            log_path = os.path.join(args.log_dir_path, log_path)
        log_handler = logging.FileHandler(log_path)
        logger.addHandler(log_handler)
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
        if args.srtin[0] is None:
            logger.info('unsynchronized subtitle file not specified; skipping synchronization')
            return result
    sync_was_successful = try_sync(args, reference_pipe, result)
    if log_handler is not None and log_path is not None:
        assert args.make_test_case
        log_handler.close()
        logger.removeHandler(log_handler)
        try:
            result['retval'] += make_test_case(args, npy_savename, sync_was_successful)
        finally:
            os.remove(log_path)
    return result


def add_main_args_for_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        'reference',
        help='Reference (video, subtitles, or a numpy array with VAD speech) to which to synchronize input subtitles.'
    )
    parser.add_argument('-i', '--srtin', nargs='*', help='Input subtitles file (default=stdin).')
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
             'formatted according to ffmpeg conventions. For example, 0:s:0 '
             'uses the first subtitle track; 0:a:3 would use the third audio track. '
             'You can also drop the leading `0:`; i.e. use s:0 or a:3, respectively. '
             'Example: `ffs ref.mkv -i in.srt -o out.srt --reference-stream s:2`'
    )


def add_cli_only_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument('--max-offset-seconds', type=float, default=DEFAULT_MAX_OFFSET_SECONDS,
                        help='The max allowed offset seconds for any subtitle segment '
                             '(default=%d seconds).' % DEFAULT_MAX_OFFSET_SECONDS)
    parser.add_argument('--apply-offset-seconds', type=float, default=DEFAULT_APPLY_OFFSET_SECONDS,
                        help='Apply a predefined offset in seconds to all subtitle segments '
                             '(default=%d seconds).' % DEFAULT_APPLY_OFFSET_SECONDS)
    parser.add_argument('--frame-rate', type=int, default=DEFAULT_FRAME_RATE,
                        help='Frame rate for audio extraction (default=%d).' % DEFAULT_FRAME_RATE)
    parser.add_argument('--skip-infer-framerate-ratio', action='store_true',
                        help='If set, do not try to infer framerate ratio based on duration ratio.')
    parser.add_argument('--non-speech-label', type=float, default=DEFAULT_NON_SPEECH_LABEL,
                        help='Label to use for frames detected as non-speech (default=%f)' % DEFAULT_NON_SPEECH_LABEL)
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
    parser.add_argument('--log-dir-path', default=None, help='Where to save ffsubsync.log file (must be an existing '
                        'directory).')
    parser.add_argument('--vlc-mode', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--gui-mode', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--skip-sync', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--gss', action='store_true', help=argparse.SUPPRESS)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Synchronize subtitles with video.')
    add_main_args_for_cli(parser)
    add_cli_only_args(parser)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    return run(args)['retval']


if __name__ == "__main__":
    sys.exit(main())
