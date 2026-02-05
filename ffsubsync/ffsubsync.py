#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import cast, Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np

from ffsubsync.aligners import FFTAligner, MaxScoreAligner, compute_weighted_median_offset
from ffsubsync.constants import (
    DEFAULT_APPLY_OFFSET_SECONDS,
    DEFAULT_FRAME_RATE,
    DEFAULT_MAX_OFFSET_SECONDS,
    DEFAULT_MAX_SUBTITLE_SECONDS,
    DEFAULT_NON_SPEECH_LABEL,
    DEFAULT_START_SECONDS,
    DEFAULT_VAD,
    DEFAULT_ENCODING,
    FRAMERATE_RATIOS,
    SAMPLE_RATE,
    SUBTITLE_EXTENSIONS,
    REMOTE_URL_PROTOCOLS,
    is_remote_url,
    DEFAULT_MIN_SCORE,
    DEFAULT_QUALITY_MAX_OFFSET_SECONDS,
    DEFAULT_MAX_FRAMERATE_DEVIATION,
)
from ffsubsync.ffmpeg_utils import ffmpeg_bin_path
from ffsubsync.sklearn_shim import Pipeline, TransformerMixin
from ffsubsync.speech_transformers import (
    VideoSpeechTransformer,
    DeserializeSpeechTransformer,
    make_subtitle_speech_pipeline,
)
from ffsubsync.subtitle_parser import make_subtitle_parser
from ffsubsync.subtitle_transformers import SubtitleMerger, SubtitleShifter
from ffsubsync.version import get_version


logger: logging.Logger = logging.getLogger(__name__)


def validate_remote_url(url: str, timeout: int = 10) -> bool:
    """Check if a remote URL is accessible.
    
    Tries HEAD request first, falls back to GET request (reading minimal data) if not supported.
    
    Args:
        url: The URL to check.
        timeout: Timeout in seconds.
        
    Returns:
        True if URL is accessible, False otherwise.
    """
    import urllib.request
    import urllib.error
    
    # Only pre-check http/https, skip other protocols (rtmp, etc.)
    if not url.startswith(('http://', 'https://')):
        return True
    
    try:
        # Try HEAD request first
        req = urllib.request.Request(url, method='HEAD')
        req.add_header('User-Agent', 'ffsubsync')
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError):
        pass
    except Exception:
        pass
    
    # Fallback: use GET request, read minimal data to verify accessibility
    try:
        req = urllib.request.Request(url, method='GET')
        req.add_header('User-Agent', 'ffsubsync')
        with urllib.request.urlopen(req, timeout=timeout) as response:
            # Read minimal data to verify connection
            response.read(1024)
            return True
    except Exception:
        return False


def override(args: argparse.Namespace, **kwargs: Any) -> Dict[str, Any]:
    args_dict = dict(args.__dict__)
    args_dict.update(kwargs)
    return args_dict


def _ref_format(ref_fname: Optional[str]) -> Optional[str]:
    if ref_fname is None:
        return None
    return ref_fname[-3:]


def make_test_case(
    args: argparse.Namespace, npy_savename: Optional[str], sync_was_successful: bool
) -> int:
    if npy_savename is None:
        raise ValueError("need non-null npy_savename")
    # Handle directory name generation for remote URL (URL cannot be used directly as directory name)
    if is_remote_url(args.reference):
        parsed = urlparse(args.reference)
        ref_name = os.path.basename(parsed.path) or parsed.netloc.replace('.', '_')
    else:
        ref_name = args.reference
    tar_dir = "{}.{}".format(
        ref_name, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    logger.info("creating test archive {}.tar.gz...".format(tar_dir))
    os.mkdir(tar_dir)
    try:
        log_path = "ffsubsync.log"
        if args.log_dir_path is not None and os.path.isdir(args.log_dir_path):
            log_path = os.path.join(args.log_dir_path, log_path)
        shutil.copy(log_path, tar_dir)
        shutil.copy(args.srtin[0], tar_dir)
        if sync_was_successful:
            shutil.move(args.srtout, tar_dir)
        if _ref_format(args.reference) in SUBTITLE_EXTENSIONS:
            # Remote URL subtitles cannot be copied directly, show warning
            if is_remote_url(args.reference):
                logger.warning("Remote URL reference cannot be included in test case archive")
            else:
                shutil.copy(args.reference, tar_dir)
        elif args.serialize_speech or args.reference == npy_savename:
            shutil.copy(npy_savename, tar_dir)
        else:
            shutil.move(npy_savename, tar_dir)
        supported_formats = set(list(zip(*shutil.get_archive_formats()))[0])
        preferred_formats = ["gztar", "bztar", "xztar", "zip", "tar"]
        for archive_format in preferred_formats:
            if archive_format in supported_formats:
                shutil.make_archive(tar_dir, archive_format, os.curdir, tar_dir)
                break
        else:
            logger.error(
                "failed to create test archive; no formats supported "
                "(this should not happen)"
            )
            return 1
        logger.info("...done")
    finally:
        shutil.rmtree(tar_dir)
    return 0


def get_srt_pipe_maker(
    args: argparse.Namespace, srtin: Optional[str]
) -> Callable[[Optional[float]], Union[Pipeline, Callable[[float], Pipeline]]]:
    if srtin is None:
        srtin_format = "srt"
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
        framerate_ratios = list(
            np.concatenate(
                [np.array(FRAMERATE_RATIOS), 1.0 / np.array(FRAMERATE_RATIOS)]
            )
        )
        if args.gss:
            framerate_ratios.append(None)
        return framerate_ratios


def get_adaptive_thresholds(
    video_duration: float,
    args: argparse.Namespace,
    total_speech_frames: Optional[float] = None
) -> Dict[str, float]:
    """Calculate adaptive quality thresholds based on video duration and speech density.
    
    Args:
        video_duration: Video duration in seconds
        args: Command line arguments containing base thresholds
        total_speech_frames: Total speech frames detected (optional, for density adjustment)
    
    Returns:
        Dictionary with adjusted thresholds: min_score, max_offset, max_framerate_dev
    """
    base_min_score = getattr(args, 'min_score', 0.0)
    base_max_offset = getattr(args, 'quality_max_offset_seconds', 30.0)
    base_max_framerate_dev = getattr(args, 'max_framerate_deviation', 0.05)
    
    # Duration-based adjustments
    if video_duration < 300:  # < 5 minutes
        score_multiplier = 0.5
        offset_multiplier = 0.3
        framerate_multiplier = 1.5
    elif video_duration < 1800:  # < 30 minutes
        score_multiplier = 0.8
        offset_multiplier = 0.5
        framerate_multiplier = 1.2
    else:
        score_multiplier = 1.0
        offset_multiplier = 1.0
        framerate_multiplier = 1.0
    
    # Calculate adjusted thresholds
    adjusted_min_score = base_min_score * score_multiplier
    adjusted_max_offset = min(
        base_max_offset * offset_multiplier,
        video_duration * 0.25  # Never exceed 25% of video duration
    )
    adjusted_max_framerate_dev = base_max_framerate_dev * framerate_multiplier
    
    # Speech density adjustment (if available)
    if total_speech_frames is not None and video_duration > 0:
        speech_density = total_speech_frames / (video_duration * SAMPLE_RATE)
        if speech_density < 0.3:  # Low speech density
            adjusted_min_score *= 0.5
            adjusted_max_offset *= 1.5
    
    return {
        'min_score': adjusted_min_score,
        'max_offset': adjusted_max_offset,
        'max_framerate_dev': adjusted_max_framerate_dev
    }


def try_multi_segment_sync(
    args: argparse.Namespace, result: Dict[str, Any]
) -> bool:
    """Perform multi-segment sync for remote URLs.
    
    Samples multiple segments from video, computes alignment for each,
    and returns weighted median offset.
    """
    import ffmpeg
    from ffsubsync.aligners import FailedToFindAlignmentException
    
    result["sync_was_successful"] = False
    sync_was_successful = True
    
    # Get video duration first
    try:
        probe_result = ffmpeg.probe(
            args.reference,
            cmd=ffmpeg_bin_path("ffprobe", args.gui_mode, ffmpeg_resources_path=args.ffmpeg_path),
        )
        total_duration = float(probe_result["format"]["duration"])
    except Exception as e:
        logger.error("Failed to probe video duration: %s", e)
        logger.info("Falling back to single-segment sync...")
        return try_sync(args, make_reference_pipe(args).fit(args.reference), result)
    
    logger.info("Video duration: %.1f seconds", total_duration)
    
    # Validate total_duration
    if total_duration <= 0:
        logger.error("Invalid video duration: %.1f seconds", total_duration)
        logger.info("Falling back to single-segment sync...")
        return try_sync(args, make_reference_pipe(args).fit(args.reference), result)
    
    # Calculate segment positions using VideoSpeechTransformer's method
    # Create a temporary transformer to use its segment calculation logic
    vad = args.vad or DEFAULT_VAD
    segment_calculator = VideoSpeechTransformer(
        vad=vad,
        sample_rate=SAMPLE_RATE,
        frame_rate=args.frame_rate,
        non_speech_label=args.non_speech_label,
        segment_count=getattr(args, 'segment_count', 8),
        skip_intro_outro=getattr(args, 'skip_intro_outro', False),
    )
    segments = segment_calculator._calculate_segment_positions(total_duration)
    
    if len(segments) < 2:
        logger.warning("Video too short for multi-segment sync, using single-segment...")
        return try_sync(args, make_reference_pipe(args).fit(args.reference), result)
    
    logger.info("Processing %d segments: %s", len(segments), 
               [(s['start'], s['start'] + s['duration']) for s in segments])
    
    if not args.srtin:
        args.srtin = [None]
    
    def _cleanup_temp_files(temp_files: List[str]) -> None:
        """Helper to cleanup temp files."""
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass
    
    for srtin in args.srtin:
        temp_files: List[str] = []
        try:
            srtout = srtin if args.overwrite_input else args.srtout
            srt_pipe_maker = get_srt_pipe_maker(args, srtin)
            
            # Fit subtitle pipe once
            srt_pipe = cast(Pipeline, srt_pipe_maker(1.0))
            srt_pipe.fit(srtin)
            subtitle_speech = srt_pipe.transform(srtin)
            
            segment_results = []
            
            # Define segment extraction function for parallel execution
            def extract_segment(seg: Dict) -> Tuple[int, Optional[str]]:
                """Extract a single segment, returns (index, temp_path or None)."""
                fd, temp_path = tempfile.mkstemp(suffix='.wav')
                os.close(fd)
                
                ffmpeg_args = [
                    ffmpeg_bin_path("ffmpeg", args.gui_mode, ffmpeg_resources_path=args.ffmpeg_path),
                    "-loglevel", "warning",
                    "-ss", str(seg['start']),
                    "-i", args.reference,
                    "-t", str(seg['duration']),
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", str(args.frame_rate),
                    "-ac", "1",
                    "-y", temp_path
                ]
                
                # Dynamic timeout: base 60s + 2s per segment second (for slow networks)
                segment_timeout = 60 + seg['duration'] * 2
                max_retries = 2
                
                for retry in range(max_retries):
                    try:
                        subprocess.run(ffmpeg_args, capture_output=True, check=True, timeout=segment_timeout)
                        return seg['index'], temp_path
                    except subprocess.TimeoutExpired:
                        logger.warning("Timeout extracting segment %d (attempt %d/%d)", 
                                      seg['index'], retry + 1, max_retries)
                    except subprocess.CalledProcessError as e:
                        logger.warning("Failed to extract segment %d (attempt %d/%d): %s", 
                                      seg['index'], retry + 1, max_retries, 
                                      e.stderr[:200] if e.stderr else str(e))
                    except Exception as e:
                        logger.warning("Error extracting segment %d (attempt %d/%d): %s", 
                                      seg['index'], retry + 1, max_retries, e)
                
                # Cleanup on failure
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                logger.warning("Giving up on segment %d after %d attempts", seg['index'], max_retries)
                return seg['index'], None
            
            # Parallel extraction of all segments
            parallel_workers = min(getattr(args, 'parallel_workers', 4), len(segments))
            logger.info("Extracting %d segments in parallel (workers=%d)...", len(segments), parallel_workers)
            
            extracted_segments: Dict[int, str] = {}
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {executor.submit(extract_segment, seg): seg for seg in segments}
                for future in as_completed(futures):
                    seg = futures[future]
                    try:
                        idx, temp_path = future.result()
                        if temp_path:
                            extracted_segments[idx] = temp_path
                            temp_files.append(temp_path)
                            logger.info("  Segment %d extracted successfully", idx)
                    except Exception as e:
                        logger.warning("Unexpected error extracting segment %d: %s", seg['index'], e)
            
            logger.info("Extraction complete: %d/%d segments successful", len(extracted_segments), len(segments))
            
            # Process extracted segments sequentially (VAD + alignment)
            for seg in segments:
                if seg['index'] not in extracted_segments:
                    continue
                    
                temp_path = extracted_segments[seg['index']]
                logger.info("Processing segment %d/%d (start=%ds)...", 
                           seg['index'] + 1, len(segments), seg['start'])
                
                # Create a VideoSpeechTransformer for this segment
                vad = args.vad or DEFAULT_VAD
                segment_transformer = VideoSpeechTransformer(
                    vad=vad,
                    sample_rate=SAMPLE_RATE,
                    frame_rate=args.frame_rate,
                    non_speech_label=args.non_speech_label,
                    start_seconds=0,
                    ffmpeg_path=args.ffmpeg_path,
                    vlc_mode=args.vlc_mode,
                    gui_mode=args.gui_mode,
                )
                
                try:
                    segment_transformer.fit(temp_path)
                    ref_speech = segment_transformer.transform(temp_path)
                    
                    # Compute alignment for this segment
                    # Adjust subtitle speech to match segment time range
                    seg_start_frame = int(seg['start'] * SAMPLE_RATE)
                    seg_end_frame = int((seg['start'] + seg['duration']) * SAMPLE_RATE)
                    
                    # Ensure we don't exceed bounds
                    if seg_start_frame >= len(subtitle_speech):
                        logger.warning("Segment %d beyond subtitle range, skipping", seg['index'])
                        continue
                    
                    seg_end_frame = min(seg_end_frame, len(subtitle_speech))
                    sub_segment = subtitle_speech[seg_start_frame:seg_end_frame]
                    
                    if len(sub_segment) == 0 or len(ref_speech) == 0:
                        logger.warning("Empty segment %d, skipping", seg['index'])
                        continue
                    
                    # Align
                    max_offset = args.max_offset_seconds if args.max_offset_seconds else DEFAULT_MAX_OFFSET_SECONDS
                    aligner = FFTAligner(max_offset_samples=int(max_offset * SAMPLE_RATE))
                    aligner.fit(ref_speech, sub_segment, get_score=True)
                    score, offset_samples = aligner.transform()
                    
                    offset_seconds = offset_samples / float(SAMPLE_RATE)
                    
                    segment_results.append({
                        'segment_start': seg['start'],
                        'segment_index': seg['index'],
                        'offset_seconds': offset_seconds,
                        'offset_samples': offset_samples,
                        'score': score,
                    })
                    
                    logger.info("  Segment %d: offset=%.3fs, score=%.3f", 
                               seg['index'], offset_seconds, score)
                    
                except Exception as e:
                    logger.warning("Failed to process segment %d: %s", seg['index'], e)
                    continue
            
            # Compute weighted median offset
            if len(segment_results) == 0:
                logger.warning("No valid segments, falling back to single-segment sync")
                return try_sync(args, make_reference_pipe(args).fit(args.reference), result)
            
            if len(segment_results) == 1:
                # Use the single valid segment directly instead of re-processing
                logger.info("Only 1 valid segment, using its result directly")
                offset_seconds = segment_results[0]['offset_seconds']
                best_score = segment_results[0]['score']
                valid_segments = segment_results
            else:
                try:
                    offset_seconds, best_score, valid_segments = compute_weighted_median_offset(
                        segment_results,
                        min_score=VideoSpeechTransformer.MIN_SEGMENT_SCORE
                    )
                except FailedToFindAlignmentException as e:
                    logger.error("Multi-segment alignment failed: %s", e)
                    sync_was_successful = False
                    continue
            
            offset_seconds += args.apply_offset_seconds
            
            logger.info("Final offset: %.3f seconds (from %d segments)", offset_seconds, len(valid_segments))
            logger.info("Average score: %.3f", best_score)
            
            # Apply offset
            scale_step = srt_pipe.named_steps["scale"]
            output_steps: List[Tuple[str, TransformerMixin]] = [
                ("shift", SubtitleShifter(offset_seconds))
            ]
            output_pipe = Pipeline(output_steps)
            out_subs = output_pipe.fit_transform(scale_step.subs_)
            
            if args.output_encoding != "same":
                out_subs = out_subs.set_encoding(args.output_encoding)
            
            suppress_output_thresh = args.suppress_output_if_offset_less_than
            if offset_seconds >= (suppress_output_thresh or float("-inf")):
                logger.info("writing output to {}".format(srtout or "stdout"))
                out_subs.write_file(srtout)
            else:
                logger.warning(
                    "suppressing output because offset %s was less than suppression threshold %s",
                    offset_seconds, suppress_output_thresh
                )
            
            result["offset_seconds"] = offset_seconds
            result["segment_count"] = len(valid_segments)
            
        except Exception:
            sync_was_successful = False
            logger.exception("failed to sync %s", srtin)
        finally:
            # Ensure temp files are cleaned up even on exception
            _cleanup_temp_files(temp_files)
    
    result["sync_was_successful"] = sync_was_successful
    return sync_was_successful


def try_sync(
    args: argparse.Namespace, reference_pipe: Optional[Pipeline], result: Dict[str, Any]
) -> bool:
    result["sync_was_successful"] = False
    sync_was_successful = True
    logger.info(
        "extracting speech segments from %s...",
        "stdin" if not args.srtin else "subtitles file(s) {}".format(args.srtin),
    )
    if not args.srtin:
        args.srtin = [None]
    for srtin in args.srtin:
        try:
            skip_sync = args.skip_sync or reference_pipe is None
            skip_infer_framerate_ratio = (
                args.skip_infer_framerate_ratio or reference_pipe is None
            )
            srtout = srtin if args.overwrite_input else args.srtout
            srt_pipe_maker = get_srt_pipe_maker(args, srtin)
            framerate_ratios = get_framerate_ratios_to_try(args)
            srt_pipes = [srt_pipe_maker(1.0)] + [
                srt_pipe_maker(rat) for rat in framerate_ratios
            ]
            for srt_pipe in srt_pipes:
                if callable(srt_pipe):
                    continue
                else:
                    srt_pipe.fit(srtin)
            if not skip_infer_framerate_ratio and hasattr(
                reference_pipe[-1], "num_frames"
            ):
                inferred_framerate_ratio_from_length = (
                    float(reference_pipe[-1].num_frames)
                    / cast(Pipeline, srt_pipes[0])[-1].num_frames
                )
                logger.info(
                    "inferred frameratio ratio: %.3f"
                    % inferred_framerate_ratio_from_length
                )
                srt_pipes.append(
                    cast(
                        Pipeline, srt_pipe_maker(inferred_framerate_ratio_from_length)
                    ).fit(srtin)
                )
                logger.info("...done")
            logger.info("computing alignments...")
            if skip_sync:
                best_score = 0.0
                best_srt_pipe = cast(Pipeline, srt_pipes[0])
                offset_samples = 0
            else:
                (best_score, offset_samples), best_srt_pipe = MaxScoreAligner(
                    FFTAligner, srtin, SAMPLE_RATE, args.max_offset_seconds
                ).fit_transform(
                    reference_pipe.transform(args.reference),
                    srt_pipes,
                )
            if best_score < 0:
                sync_was_successful = False
            logger.info("...done")
            offset_seconds = (
                offset_samples / float(SAMPLE_RATE) + args.apply_offset_seconds
            )
            scale_step = best_srt_pipe.named_steps["scale"]
            logger.info("score: %.3f", best_score)
            logger.info("offset seconds: %.3f", offset_seconds)
            logger.info("framerate scale factor: %.3f", scale_step.scale_factor)
            
            # Quality protection: check if alignment quality is too low
            skip_sync_due_to_quality = False
            quality_reasons = []
            if getattr(args, 'skip_sync_on_low_quality', False):
                # Get thresholds (adaptive or fixed)
                if getattr(args, 'adaptive_thresholds', False):
                    # Try to get video duration and speech frames for adaptive thresholds
                    video_duration = result.get('video_duration', 0)
                    total_speech_frames = result.get('total_speech_frames', None)
                    thresholds = get_adaptive_thresholds(video_duration, args, total_speech_frames)
                    min_score_thresh = thresholds['min_score']
                    max_offset_thresh = thresholds['max_offset']
                    max_framerate_dev_thresh = thresholds['max_framerate_dev']
                    logger.info(
                        "Using adaptive thresholds: min_score=%.1f, max_offset=%.1fs, max_framerate_dev=%.3f",
                        min_score_thresh, max_offset_thresh, max_framerate_dev_thresh
                    )
                else:
                    min_score_thresh = args.min_score
                    max_offset_thresh = args.quality_max_offset_seconds
                    max_framerate_dev_thresh = args.max_framerate_deviation
                
                if best_score < min_score_thresh:
                    quality_reasons.append(f"score {best_score:.1f} < {min_score_thresh}")
                if abs(offset_seconds) > max_offset_thresh:
                    quality_reasons.append(f"|offset| {abs(offset_seconds):.1f}s > {max_offset_thresh}s")
                framerate_deviation = abs(scale_step.scale_factor - 1.0)
                if framerate_deviation > max_framerate_dev_thresh:
                    quality_reasons.append(f"framerate deviation {framerate_deviation:.3f} > {max_framerate_dev_thresh}")
                
                if quality_reasons:
                    skip_sync_due_to_quality = True
                    logger.warning(
                        "Low quality alignment detected, outputting original subtitles. Reasons: %s",
                        "; ".join(quality_reasons)
                    )
            
            if skip_sync_due_to_quality:
                # Output original subtitles without modification
                output_steps = []  # No shift, no scale
                out_subs = scale_step.subs_.clone_props_for_subs(list(scale_step.subs_))
                sync_was_successful = False
            else:
                output_steps: List[Tuple[str, TransformerMixin]] = [
                    ("shift", SubtitleShifter(offset_seconds))
                ]
                if args.merge_with_reference:
                    output_steps.append(
                        ("merge", SubtitleMerger(reference_pipe.named_steps["parse"].subs_))
                    )
                output_pipe = Pipeline(output_steps)
                out_subs = output_pipe.fit_transform(scale_step.subs_)
            if args.output_encoding != "same":
                out_subs = out_subs.set_encoding(args.output_encoding)
            suppress_output_thresh = args.suppress_output_if_offset_less_than
            if offset_seconds >= (suppress_output_thresh or float("-inf")):
                logger.info("writing output to {}".format(srtout or "stdout"))
                out_subs.write_file(srtout)
            else:
                logger.warning(
                    "suppressing output because offset %s was less than suppression threshold %s",
                    offset_seconds,
                    args.suppress_output_if_offset_less_than,
                )
        except Exception:
            sync_was_successful = False
            logger.exception("failed to sync %s", srtin)
        else:
            result["offset_seconds"] = offset_seconds
            result["framerate_scale_factor"] = scale_step.scale_factor
    result["sync_was_successful"] = sync_was_successful
    return sync_was_successful


def make_reference_pipe(args: argparse.Namespace) -> Pipeline:
    ref_format = _ref_format(args.reference)
    if ref_format in SUBTITLE_EXTENSIONS:
        if args.vad is not None:
            logger.warning("Vad specified, but reference was not a movie")
        return cast(
            Pipeline,
            make_subtitle_speech_pipeline(
                fmt=ref_format,
                **override(args, encoding=args.reference_encoding or DEFAULT_ENCODING),
            ),
        )
    elif ref_format in ("npy", "npz"):
        if args.vad is not None:
            logger.warning("Vad specified, but reference was not a movie")
        return Pipeline(
            [("deserialize", DeserializeSpeechTransformer(args.non_speech_label))]
        )
    else:
        vad = args.vad or DEFAULT_VAD
        if args.reference_encoding is not None:
            logger.warning(
                "Reference srt encoding specified, but reference was a video file"
            )
        ref_stream = args.reference_stream
        if ref_stream is not None and not ref_stream.startswith("0:"):
            ref_stream = "0:" + ref_stream
        return Pipeline(
            [
                (
                    "speech_extract",
                    VideoSpeechTransformer(
                        vad=vad,
                        sample_rate=SAMPLE_RATE,
                        frame_rate=args.frame_rate,
                        non_speech_label=args.non_speech_label,
                        start_seconds=args.start_seconds,
                        ffmpeg_path=args.ffmpeg_path,
                        ref_stream=ref_stream,
                        vlc_mode=args.vlc_mode,
                        gui_mode=args.gui_mode,
                        extract_audio_first=getattr(args, 'extract_audio_first', False),
                        max_duration_seconds=getattr(args, 'max_duration_seconds', None),
                        multi_segment_sync=getattr(args, 'multi_segment_sync', False),
                        segment_count=getattr(args, 'segment_count', 8),
                    ),
                ),
            ]
        )


def extract_subtitles_from_reference(args: argparse.Namespace) -> int:
    stream = args.extract_subs_from_stream
    if not stream.startswith("0:s:"):
        stream = "0:s:{}".format(stream)
    elif not stream.startswith("0:") and stream.startswith("s:"):
        stream = "0:{}".format(stream)
    if not stream.startswith("0:s:"):
        logger.error(
            "invalid stream for subtitle extraction: %s", args.extract_subs_from_stream
        )
    ffmpeg_args = [
        ffmpeg_bin_path("ffmpeg", args.gui_mode, ffmpeg_resources_path=args.ffmpeg_path)
    ]
    ffmpeg_args.extend(
        [
            "-y",
            "-nostdin",
            "-loglevel",
            "fatal",
            "-i",
            args.reference,
            "-map",
            "{}".format(stream),
            "-f",
            "srt",
        ]
    )
    if args.srtout is None:
        ffmpeg_args.append("-")
    else:
        ffmpeg_args.append(args.srtout)
    logger.info(
        "attempting to extract subtitles to {} ...".format(
            "stdout" if args.srtout is None else args.srtout
        )
    )
    retcode = subprocess.call(ffmpeg_args)
    if retcode == 0:
        logger.info("...done")
    else:
        logger.error(
            "ffmpeg unable to extract subtitles from reference; return code %d", retcode
        )
    return retcode


def validate_args(args: argparse.Namespace) -> None:
    if args.vlc_mode:
        logger.setLevel(logging.CRITICAL)
    if args.reference is None:
        if args.apply_offset_seconds == 0 or not args.srtin:
            raise ValueError(
                "`reference` required unless `--apply-offset-seconds` specified"
            )
    if args.apply_offset_seconds != 0:
        if not args.srtin:
            args.srtin = [args.reference]
        if not args.srtin:
            raise ValueError(
                "at least one of `srtin` or `reference` must be specified to apply offset seconds"
            )
    if args.srtin:
        if len(args.srtin) > 1 and not args.overwrite_input:
            raise ValueError(
                "cannot specify multiple input srt files without overwriting"
            )
        if len(args.srtin) > 1 and args.make_test_case:
            raise ValueError("cannot specify multiple input srt files for test cases")
        if len(args.srtin) > 1 and args.gui_mode:
            raise ValueError("cannot specify multiple input srt files in GUI mode")
    if (
        args.make_test_case and not args.gui_mode
    ):  # this validation not necessary for gui mode
        if not args.srtin or args.srtout is None:
            raise ValueError(
                "need to specify input and output srt files for test cases"
            )
    if args.overwrite_input:
        if args.extract_subs_from_stream is not None:
            raise ValueError(
                "input overwriting not allowed for extracting subtitles from reference"
            )
        if not args.srtin:
            raise ValueError(
                "need to specify input srt if --overwrite-input "
                "is specified since we cannot overwrite stdin"
            )
        if args.srtout is not None:
            raise ValueError(
                "overwrite input set but output file specified; "
                "refusing to run in case this was not intended"
            )
    if args.extract_subs_from_stream is not None:
        if args.make_test_case:
            raise ValueError("test case is for sync and not subtitle extraction")
        if args.srtin:
            raise ValueError(
                "stream specified for reference subtitle extraction; "
                "-i flag for sync input not allowed"
            )


def validate_file_permissions(args: argparse.Namespace) -> None:
    error_string_template = (
        "unable to {action} {file}; "
        "try ensuring file exists and has correct permissions"
    )
    # Remote URLs are checked for accessibility, local files are checked for permissions
    if args.reference is not None:
        if is_remote_url(args.reference):
            if not validate_remote_url(args.reference):
                raise ValueError(
                    "unable to access remote URL: {}; "
                    "please check the URL is correct and accessible".format(args.reference)
                )
        elif not os.access(args.reference, os.R_OK):
            raise ValueError(
                error_string_template.format(action="read reference", file=args.reference)
            )
    if args.srtin:
        for srtin in args.srtin:
            if srtin is not None and not os.access(srtin, os.R_OK):
                raise ValueError(
                    error_string_template.format(
                        action="read input subtitles", file=srtin
                    )
                )
    if (
        args.srtout is not None
        and os.path.exists(args.srtout)
        and not os.access(args.srtout, os.W_OK)
    ):
        raise ValueError(
            error_string_template.format(
                action="write output subtitles", file=args.srtout
            )
        )
    if args.make_test_case or args.serialize_speech:
        npy_savename = _npy_savename(args)
        if os.path.exists(npy_savename) and not os.access(npy_savename, os.W_OK):
            raise ValueError(
                "unable to write test case file archive %s (try checking permissions)"
                % npy_savename
            )


def _setup_logging(
    args: argparse.Namespace,
) -> Tuple[Optional[str], Optional[logging.FileHandler]]:
    log_handler = None
    log_path = None
    if args.make_test_case or args.log_dir_path is not None:
        log_path = "ffsubsync.log"
        if args.log_dir_path is not None and os.path.isdir(args.log_dir_path):
            log_path = os.path.join(args.log_dir_path, log_path)
        log_handler = logging.FileHandler(log_path)
        logger.addHandler(log_handler)
        logger.info("this log will be written to %s", os.path.abspath(log_path))
    return log_path, log_handler


def _npy_savename(args: argparse.Namespace) -> str:
    """Generate NPZ save filename.
    
    Handles both local paths and remote URLs. For URLs, extracts filename;
    if extraction fails, uses domain + timestamp as fallback.
    """
    if is_remote_url(args.reference):
        parsed = urlparse(args.reference)
        # Extract filename from URL path, remove extension
        base_name = os.path.splitext(os.path.basename(parsed.path))[0]
        # If URL path is empty or just '/', use domain + timestamp as base name (prevent overwriting)
        if not base_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = parsed.netloc.replace('.', '_') + "_" + timestamp
        return base_name + ".npz"
    return os.path.splitext(args.reference)[0] + ".npz"


def _run_impl(args: argparse.Namespace, result: Dict[str, Any]) -> bool:
    if args.extract_subs_from_stream is not None:
        result["retval"] = extract_subtitles_from_reference(args)
        return True
    if args.srtin is not None and (
        args.reference is None
        or (len(args.srtin) == 1 and args.srtin[0] == args.reference)
    ):
        return try_sync(args, None, result)
    
    # Use multi-segment sync for remote URLs when enabled
    if getattr(args, 'multi_segment_sync', False) and is_remote_url(args.reference):
        logger.info("Using multi-segment sync mode for remote URL...")
        return try_multi_segment_sync(args, result)
    
    reference_pipe = make_reference_pipe(args)
    logger.info("extracting speech segments from reference '%s'...", args.reference)
    reference_pipe.fit(args.reference)
    logger.info("...done")
    
    # Store video metadata for adaptive thresholds
    if hasattr(reference_pipe[-1], 'video_speech_results_'):
        speech_results = reference_pipe[-1].video_speech_results_
        result['total_speech_frames'] = float(np.sum(speech_results))
        result['video_duration'] = len(speech_results) / float(SAMPLE_RATE)
    if args.make_test_case or args.serialize_speech:
        logger.info("serializing speech...")
        np.savez_compressed(
            _npy_savename(args), speech=reference_pipe.transform(args.reference)
        )
        logger.info("...done")
        if not args.srtin:
            logger.info(
                "unsynchronized subtitle file not specified; skipping synchronization"
            )
            return False
    return try_sync(args, reference_pipe, result)


def validate_and_transform_args(
    parser_or_args: Union[argparse.ArgumentParser, argparse.Namespace]
) -> Optional[argparse.Namespace]:
    if isinstance(parser_or_args, argparse.Namespace):
        parser = None
        args = parser_or_args
    else:
        parser = parser_or_args
        args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as e:
        logger.error(e)
        if parser is not None:
            parser.print_usage()
        return None
    if args.gui_mode and args.srtout is None:
        args.srtout = "{}.synced.srt".format(os.path.splitext(args.srtin[0])[0])
    try:
        validate_file_permissions(args)
    except ValueError as e:
        logger.error(e)
        return None
    ref_format = _ref_format(args.reference)
    if args.merge_with_reference and ref_format not in SUBTITLE_EXTENSIONS:
        logger.error(
            "merging synced output with reference only valid "
            "when reference composed of subtitles"
        )
        return None
    return args


def run(
    parser_or_args: Union[argparse.ArgumentParser, argparse.Namespace]
) -> Dict[str, Any]:
    sync_was_successful = False
    result = {
        "retval": 0,
        "offset_seconds": None,
        "framerate_scale_factor": None,
    }
    args = validate_and_transform_args(parser_or_args)
    if args is None:
        result["retval"] = 1
        return result
    log_path, log_handler = _setup_logging(args)
    try:
        sync_was_successful = _run_impl(args, result)
        result["sync_was_successful"] = sync_was_successful
        return result
    finally:
        if log_handler is not None and log_path is not None:
            log_handler.close()
            logger.removeHandler(log_handler)
            if args.make_test_case:
                result["retval"] += make_test_case(
                    args, _npy_savename(args), sync_was_successful
                )
            if args.log_dir_path is None or not os.path.isdir(args.log_dir_path):
                os.remove(log_path)


def add_main_args_for_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "reference",
        nargs="?",
        help=(
            "Reference (video, subtitles, or a numpy array with VAD speech) "
            "to which to synchronize input subtitles."
        ),
    )
    parser.add_argument(
        "-i", "--srtin", nargs="*", help="Input subtitles file (default=stdin)."
    )
    parser.add_argument(
        "-o", "--srtout", help="Output subtitles file (default=stdout)."
    )
    parser.add_argument(
        "--merge-with-reference",
        "--merge",
        action="store_true",
        help="Merge reference subtitles with synced output subtitles.",
    )
    parser.add_argument(
        "--make-test-case",
        "--create-test-case",
        action="store_true",
        help="If specified, serialize reference speech to a numpy array, "
        "and create an archive with input/output subtitles "
        "and serialized speech.",
    )
    parser.add_argument(
        "--reference-stream",
        "--refstream",
        "--reference-track",
        "--reftrack",
        default=None,
        help=(
            "Which stream/track in the video file to use as reference, "
            "formatted according to ffmpeg conventions. For example, 0:s:0 "
            "uses the first subtitle track; 0:a:3 would use the third audio track. "
            "You can also drop the leading `0:`; i.e. use s:0 or a:3, respectively. "
            "Example: `ffs ref.mkv -i in.srt -o out.srt --reference-stream s:2`"
        ),
    )


def add_cli_only_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="{package} {version}".format(
            package=__package__, version=get_version()
        ),
    )
    parser.add_argument(
        "--overwrite-input",
        action="store_true",
        help=(
            "If specified, will overwrite the input srt "
            "instead of writing the output to a new file."
        ),
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help="What encoding to use for reading input subtitles "
        "(default=%s)." % DEFAULT_ENCODING,
    )
    parser.add_argument(
        "--max-subtitle-seconds",
        type=float,
        default=DEFAULT_MAX_SUBTITLE_SECONDS,
        help="Maximum duration for a subtitle to appear on-screen "
        "(default=%.3f seconds)." % DEFAULT_MAX_SUBTITLE_SECONDS,
    )
    parser.add_argument(
        "--start-seconds",
        type=int,
        default=DEFAULT_START_SECONDS,
        help="Start time for processing "
        "(default=%d seconds)." % DEFAULT_START_SECONDS,
    )
    parser.add_argument(
        "--max-offset-seconds",
        type=float,
        default=DEFAULT_MAX_OFFSET_SECONDS,
        help="The max allowed offset seconds for any subtitle segment "
        "(default=%d seconds)." % DEFAULT_MAX_OFFSET_SECONDS,
    )
    parser.add_argument(
        "--apply-offset-seconds",
        type=float,
        default=DEFAULT_APPLY_OFFSET_SECONDS,
        help="Apply a predefined offset in seconds to all subtitle segments "
        "(default=%d seconds)." % DEFAULT_APPLY_OFFSET_SECONDS,
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=DEFAULT_FRAME_RATE,
        help="Frame rate for audio extraction (default=%d)." % DEFAULT_FRAME_RATE,
    )
    parser.add_argument(
        "--skip-infer-framerate-ratio",
        action="store_true",
        help="If set, do not try to infer framerate ratio based on duration ratio.",
    )
    parser.add_argument(
        "--non-speech-label",
        type=float,
        default=DEFAULT_NON_SPEECH_LABEL,
        help="Label to use for frames detected as non-speech (default=%f)"
        % DEFAULT_NON_SPEECH_LABEL,
    )
    parser.add_argument(
        "--output-encoding",
        default="utf-8",
        help="What encoding to use for writing output subtitles "
        '(default=utf-8). Can indicate "same" to use same '
        "encoding as that of the input.",
    )
    parser.add_argument(
        "--reference-encoding",
        help="What encoding to use for reading / writing reference subtitles "
        "(if applicable, default=infer).",
    )
    parser.add_argument(
        "--vad",
        choices=[
            "subs_then_webrtc",
            "webrtc",
            "subs_then_auditok",
            "auditok",
            "subs_then_silero",
            "silero",
        ],
        default=None,
        help="Which voice activity detector to use for speech extraction "
        "(if using video / audio as a reference, default={}).".format(DEFAULT_VAD),
    )
    parser.add_argument(
        "--no-fix-framerate",
        action="store_true",
        help="If specified, subsync will not attempt to correct a framerate "
        "mismatch between reference and subtitles.",
    )
    parser.add_argument(
        "--serialize-speech",
        action="store_true",
        help="If specified, serialize reference speech to a numpy array.",
    )
    parser.add_argument(
        "--extract-subs-from-stream",
        "--extract-subtitles-from-stream",
        default=None,
        help="If specified, do not attempt sync; instead, just extract subtitles"
        " from the specified stream using the reference.",
    )
    parser.add_argument(
        "--suppress-output-if-offset-less-than",
        type=float,
        default=None,
        help="If specified, do not produce output if offset below provided threshold.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        "--ffmpegpath",
        default=None,
        help="Where to look for ffmpeg and ffprobe. Uses the system PATH by default.",
    )
    parser.add_argument(
        "--log-dir-path",
        default=None,
        help=(
            "If provided, will save log file ffsubsync.log to this path "
            "(must be an existing directory)."
        ),
    )
    parser.add_argument(
        "--gss",
        action="store_true",
        help="If specified, use golden-section search to try to find"
        "the optimal framerate ratio between video and subtitles.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If specified, refuse to parse srt files with formatting issues.",
    )
    parser.add_argument(
        "--extract-audio-first",
        action="store_true",
        help="For remote URLs, extract audio to local temp file before processing. "
             "This can significantly speed up processing for remote videos.",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=int,
        default=None,
        help="Only process first N seconds of video for faster sync. "
             "Useful for long videos where offset is typically detectable early.",
    )
    parser.add_argument(
        "--multi-segment-sync",
        action="store_true",
        help="Enable multi-segment sync mode for remote URLs. "
             "Samples multiple segments from video and computes weighted median offset. "
             "Significantly faster for long remote videos.",
    )
    parser.add_argument(
        "--segment-count",
        type=int,
        default=8,
        help="Number of segments to sample when using --multi-segment-sync (default=8). "
             "Each segment is 60 seconds. More segments = more accurate but slower.",
    )
    parser.add_argument(
        "--skip-intro-outro",
        action="store_true",
        help="Skip intro/outro when using --multi-segment-sync. "
             "Skips first 30s and last 60s of video (intro/credits often have no dialogue).",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=4,
        help="Number of parallel workers for segment extraction in --multi-segment-sync (default=4). "
             "Higher values may speed up remote URL processing but increase network load.",
    )
    parser.add_argument(
        "--skip-sync-on-low-quality",
        action="store_true",
        help="If alignment quality is too low, output original subtitles without modification. "
             "Quality is determined by score, offset, and framerate deviation thresholds.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Minimum alignment score threshold. If score < this value and "
             "--skip-sync-on-low-quality is set, subtitles are not modified (default=%.1f)." 
             % DEFAULT_MIN_SCORE,
    )
    parser.add_argument(
        "--quality-max-offset-seconds",
        type=float,
        default=DEFAULT_QUALITY_MAX_OFFSET_SECONDS,
        help="Maximum offset threshold for quality check. If |offset| > this value and "
             "--skip-sync-on-low-quality is set, subtitles are not modified (default=%.1f seconds)."
             % DEFAULT_QUALITY_MAX_OFFSET_SECONDS,
    )
    parser.add_argument(
        "--max-framerate-deviation",
        type=float,
        default=DEFAULT_MAX_FRAMERATE_DEVIATION,
        help="Maximum framerate scale deviation from 1.0. If |scale - 1.0| > this value and "
             "--skip-sync-on-low-quality is set, subtitles are not modified (default=%.2f)."
             % DEFAULT_MAX_FRAMERATE_DEVIATION,
    )
    parser.add_argument(
        "--adaptive-thresholds",
        action="store_true",
        help="Automatically adjust quality thresholds based on video duration and speech density. "
             "Shorter videos and low speech density will use more relaxed thresholds.",
    )
    parser.add_argument("--vlc-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gui-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-sync", action="store_true", help=argparse.SUPPRESS)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronize subtitles with video.")
    add_main_args_for_cli(parser)
    add_cli_only_args(parser)
    return parser


def main() -> int:
    parser = make_parser()
    return run(parser)["retval"]


if __name__ == "__main__":
    sys.exit(main())
