# -*- coding: utf-8 -*-
import os
import tempfile
from contextlib import contextmanager
import logging
import io
import subprocess
import sys
from datetime import timedelta
from typing import cast, Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

import ffmpeg
import numpy as np
import tqdm

from ffsubsync.constants import (
    DEFAULT_ENCODING,
    DEFAULT_MAX_SUBTITLE_SECONDS,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_START_SECONDS,
    SAMPLE_RATE,
    is_remote_url,
)
from ffsubsync.ffmpeg_utils import ffmpeg_bin_path, subprocess_args
from ffsubsync.generic_subtitles import GenericSubtitle
from ffsubsync.sklearn_shim import TransformerMixin
from ffsubsync.sklearn_shim import Pipeline
from ffsubsync.subtitle_parser import make_subtitle_parser
from ffsubsync.subtitle_transformers import SubtitleScaler
from ffsubsync.subtitle_preprocessor import preprocess_subtitles


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def make_subtitle_speech_pipeline(
    fmt: str = "srt",
    encoding: str = DEFAULT_ENCODING,
    caching: bool = False,
    max_subtitle_seconds: int = DEFAULT_MAX_SUBTITLE_SECONDS,
    start_seconds: int = DEFAULT_START_SECONDS,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    parser=None,
    preprocess_subtitles: bool = False,
    **kwargs,
) -> Union[Pipeline, Callable[[float], Pipeline]]:
    if parser is None:
        parser = make_subtitle_parser(
            fmt,
            encoding=encoding,
            caching=caching,
            max_subtitle_seconds=max_subtitle_seconds,
            start_seconds=start_seconds,
            **kwargs,
        )
    assert parser.encoding == encoding
    assert parser.max_subtitle_seconds == max_subtitle_seconds
    assert parser.start_seconds == start_seconds

    def subpipe_maker(framerate_ratio):
        steps = [
            ("parse", parser),
        ]
        if preprocess_subtitles:
            steps.append(("preprocess", SubtitlePreprocessorTransformer()))
        steps.extend([
            ("scale", SubtitleScaler(framerate_ratio)),
            (
                "speech_extract",
                SubtitleSpeechTransformer(
                    sample_rate=SAMPLE_RATE,
                    start_seconds=start_seconds,
                    framerate_ratio=framerate_ratio,
                ),
            ),
        ])
        return Pipeline(steps)

    if scale_factor is None:
        return subpipe_maker
    else:
        return subpipe_maker(scale_factor)


def _make_auditok_detector(
    sample_rate: int, frame_rate: int, non_speech_label: float
) -> Callable[[bytes], np.ndarray]:
    try:
        from auditok import (
            BufferAudioSource,
            ADSFactory,
            AudioEnergyValidator,
            StreamTokenizer,
        )
    except ImportError as e:
        logger.error(
            """Error: auditok not installed!
        Consider installing it with `pip install auditok`. Note that auditok
        is GPLv3 licensed, which means that successfully importing it at
        runtime creates a derivative work that is GPLv3 licensed. For personal
        use this is fine, but note that any commercial use that relies on
        auditok must be open source as per the GPLv3!*
        *Not legal advice. Consult with a lawyer.
        """
        )
        raise e
    bytes_per_frame = 2
    frames_per_window = frame_rate // sample_rate
    validator = AudioEnergyValidator(sample_width=bytes_per_frame, energy_threshold=50)
    tokenizer = StreamTokenizer(
        validator=validator,
        min_length=0.2 * sample_rate,
        max_length=int(5 * sample_rate),
        max_continuous_silence=0.25 * sample_rate,
    )

    def _detect(asegment: bytes) -> np.ndarray:
        asource = BufferAudioSource(
            data_buffer=asegment,
            sampling_rate=frame_rate,
            sample_width=bytes_per_frame,
            channels=1,
        )
        ads = ADSFactory.ads(audio_source=asource, block_dur=1.0 / sample_rate)
        ads.open()
        tokens = tokenizer.tokenize(ads)
        length = (
            len(asegment) // bytes_per_frame + frames_per_window - 1
        ) // frames_per_window
        media_bstring = np.zeros(length + 1)
        for token in tokens:
            media_bstring[token[1]] = 1.0
            media_bstring[token[2] + 1] = non_speech_label - 1.0
        return np.clip(np.cumsum(media_bstring)[:-1], 0.0, 1.0)

    return _detect


def _make_webrtcvad_detector(
    sample_rate: int, frame_rate: int, non_speech_label: float
) -> Callable[[bytes], np.ndarray]:
    import webrtcvad

    vad = webrtcvad.Vad()
    vad.set_mode(3)  # set non-speech pruning aggressiveness from 0 to 3
    window_duration = 1.0 / sample_rate  # duration in seconds
    frames_per_window = int(window_duration * frame_rate + 0.5)
    bytes_per_frame = 2

    def _detect(asegment: bytes) -> np.ndarray:
        media_bstring = []
        failures = 0
        for start in range(0, len(asegment) // bytes_per_frame, frames_per_window):
            stop = min(start + frames_per_window, len(asegment) // bytes_per_frame)
            try:
                is_speech = vad.is_speech(
                    asegment[start * bytes_per_frame : stop * bytes_per_frame],
                    sample_rate=frame_rate,
                )
            except Exception:
                is_speech = False
                failures += 1
            # webrtcvad has low recall on mode 3, so treat non-speech as "not sure"
            media_bstring.append(1.0 if is_speech else non_speech_label)
        return np.array(media_bstring)

    return _detect


def _make_silero_detector(
    sample_rate: int, frame_rate: int, non_speech_label: float
) -> Callable[[bytes], np.ndarray]:
    import torch
    from silero_vad import load_silero_vad

    model = load_silero_vad()
    
    # Silero VAD requires specific window sizes:
    # 512 samples for 16kHz, 256 samples for 8kHz
    # frame_rate here is the audio sample rate (e.g., 48000)
    silero_sample_rate = 16000
    window_size_samples = 512  # for 16kHz
    
    # Calculate how many output frames we produce per input chunk
    # sample_rate is the VAD output rate (e.g., 100 Hz)
    # frame_rate is the audio sample rate (e.g., 48000 Hz)
    window_duration_seconds = window_size_samples / silero_sample_rate  # ~32ms per window
    output_frames_per_window = max(1, int(window_duration_seconds * sample_rate + 0.5))

    exception_logged = False

    def _detect(asegment) -> np.ndarray:
        # Convert bytes to float32 audio samples
        audio = np.frombuffer(asegment, np.int16).astype(np.float32) / (1 << 15)
        
        # Resample to 16kHz if needed (silero VAD expects 16kHz)
        if frame_rate != silero_sample_rate:
            # Simple resampling by interpolation
            original_length = len(audio)
            target_length = int(original_length * silero_sample_rate / frame_rate)
            if target_length > 0:
                indices = np.linspace(0, original_length - 1, target_length)
                audio = np.interp(indices, np.arange(original_length), audio)
        
        audio_tensor = torch.FloatTensor(audio)
        
        media_bstring = []
        
        # Process audio in windows
        for start in range(0, len(audio_tensor), window_size_samples):
            chunk = audio_tensor[start:start + window_size_samples]
            
            # Skip if chunk is too short
            if len(chunk) < window_size_samples:
                # Pad the last chunk if needed
                if len(chunk) > window_size_samples // 2:
                    chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - len(chunk)))
                else:
                    break
            
            try:
                speech_prob = model(chunk, silero_sample_rate).item()
            except Exception:
                nonlocal exception_logged
                if not exception_logged:
                    exception_logged = True
                    logger.exception("exception occurred during speech detection")
                speech_prob = 0.0
            
            # Expand to match expected output frame rate
            prob_value = 1.0 - (1.0 - speech_prob) * (1.0 - non_speech_label)
            for _ in range(output_frames_per_window):
                media_bstring.append(prob_value)
        
        # Reset model states after processing each chunk
        model.reset_states()
        
        return np.array(media_bstring)

    return _detect


def _make_fused_detector(
    sample_rate: int, frame_rate: int, non_speech_label: float,
    fusion_strategy: str = "weighted"
) -> Callable[[bytes], np.ndarray]:
    """Create a fused VAD detector combining webrtc and silero.
    
    Args:
        sample_rate: Output sample rate (e.g., 100 Hz)
        frame_rate: Audio sample rate (e.g., 48000 Hz)
        non_speech_label: Label for non-speech segments
        fusion_strategy: One of 'weighted', 'intersection', 'union'
    """
    webrtc_detector = _make_webrtcvad_detector(sample_rate, frame_rate, non_speech_label)
    silero_detector = _make_silero_detector(sample_rate, frame_rate, non_speech_label)
    
    def _detect(asegment: bytes) -> np.ndarray:
        webrtc_result = webrtc_detector(asegment)
        silero_result = silero_detector(asegment)
        
        # Align lengths (they may differ slightly)
        min_len = min(len(webrtc_result), len(silero_result))
        webrtc_result = webrtc_result[:min_len]
        silero_result = silero_result[:min_len]
        
        if fusion_strategy == "intersection":
            # Conservative: only mark as speech if both agree
            result = np.minimum(webrtc_result, silero_result)
        elif fusion_strategy == "union":
            # Aggressive: mark as speech if either detects
            result = np.maximum(webrtc_result, silero_result)
        else:  # weighted (default)
            # Silero is generally more accurate, give it higher weight
            result = 0.6 * silero_result + 0.4 * webrtc_result
        
        return result
    
    return _detect


class ComputeSpeechFrameBoundariesMixin:
    def __init__(self) -> None:
        self.start_frame_: Optional[int] = None
        self.end_frame_: Optional[int] = None

    @property
    def num_frames(self) -> Optional[int]:
        if self.start_frame_ is None or self.end_frame_ is None:
            return None
        return self.end_frame_ - self.start_frame_

    def fit_boundaries(
        self, speech_frames: np.ndarray
    ) -> "ComputeSpeechFrameBoundariesMixin":
        nz = np.nonzero(speech_frames > 0.5)[0]
        if len(nz) > 0:
            self.start_frame_ = int(np.min(nz))
            self.end_frame_ = int(np.max(nz))
        return self


class VideoSpeechTransformer(TransformerMixin):
    # Default segment duration in seconds for multi-segment sync
    SEGMENT_DURATION: int = 60
    # Minimum score threshold to consider a segment valid
    MIN_SEGMENT_SCORE: float = 0.3
    # Default margin at video start/end to skip (intro/credits often have no dialogue)
    DEFAULT_START_MARGIN: int = 30  # Skip first 30 seconds
    DEFAULT_END_MARGIN: int = 60    # Skip last 60 seconds
    
    def __init__(
        self,
        vad: str,
        sample_rate: int,
        frame_rate: int,
        non_speech_label: float,
        start_seconds: int = 0,
        ffmpeg_path: Optional[str] = None,
        ref_stream: Optional[str] = None,
        vlc_mode: bool = False,
        gui_mode: bool = False,
        extract_audio_first: bool = False,
        max_duration_seconds: Optional[int] = None,
        multi_segment_sync: bool = False,
        segment_count: int = 8,
        skip_intro_outro: bool = False,
    ) -> None:
        super(VideoSpeechTransformer, self).__init__()
        self.vad: str = vad
        self.sample_rate: int = sample_rate
        self.frame_rate: int = frame_rate
        self._non_speech_label: float = non_speech_label
        self.start_seconds: int = start_seconds
        self.ffmpeg_path: Optional[str] = ffmpeg_path
        self.ref_stream: Optional[str] = ref_stream
        self.vlc_mode: bool = vlc_mode
        self.gui_mode: bool = gui_mode
        self.extract_audio_first: bool = extract_audio_first
        self.max_duration_seconds: Optional[int] = max_duration_seconds
        self.multi_segment_sync: bool = multi_segment_sync
        self.segment_count: int = segment_count
        self.skip_intro_outro: bool = skip_intro_outro
        self.video_speech_results_: Optional[np.ndarray] = None
        self._temp_audio_file: Optional[str] = None
        # Multi-segment sync results
        self.segment_results_: Optional[List[Dict]] = None

    def _extract_audio_to_temp(self, url: str) -> str:
        """Extract audio from remote URL to local temp file."""
        # Determine audio extension from URL or use default
        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[1] or '.m4a'
        if ext not in ['.m4a', '.aac', '.mp3', '.wav', '.ogg']:
            ext = '.m4a'
        
        # Create temp file
        fd, temp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        
        logger.info("Extracting audio from remote URL to temp file...")
        
        # Get total duration first for progress bar and smart duration limit
        total_duration = None
        effective_max_duration = self.max_duration_seconds
        try:
            total_duration = float(
                ffmpeg.probe(
                    url,
                    cmd=ffmpeg_bin_path(
                        "ffprobe", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
                    ),
                )["format"]["duration"]
            )
            # Smart adjustment: if max_duration_seconds exceeds video duration, use actual duration
            if self.max_duration_seconds and total_duration:
                if self.max_duration_seconds > total_duration:
                    logger.info(
                        "max_duration_seconds (%d) exceeds video duration (%.1f), using actual duration",
                        self.max_duration_seconds, total_duration
                    )
                    effective_max_duration = int(total_duration)
                total_duration = min(total_duration, self.max_duration_seconds)
        except Exception:
            pass
        
        ffmpeg_args = [
            ffmpeg_bin_path(
                "ffmpeg", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
            ),
            "-loglevel", "info",
            "-stats",
            "-i", url,
            "-vn",  # No video
            "-acodec", "copy",  # Copy audio codec (fast)
        ]
        if effective_max_duration:
            ffmpeg_args.extend(["-t", str(effective_max_duration)])
        ffmpeg_args.extend(["-y", temp_path])
        
        try:
            process = subprocess.Popen(
                ffmpeg_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Use tqdm for progress display (consistent with project style)
            import select
            with tqdm.tqdm(
                total=total_duration,
                unit="s",
                desc="Extracting audio",
                disable=self.vlc_mode
            ) as pbar:
                current_time = 0.0
                # Timeout for reading stderr to prevent blocking
                read_timeout = 30.0  # seconds
                last_activity = time.time()
                
                while True:
                    # Check if process has finished
                    if process.poll() is not None:
                        # Read any remaining output
                        remaining = process.stderr.read()
                        if remaining and "time=" in remaining:
                            try:
                                time_str = remaining.split("time=")[-1].split()[0]
                                parts = time_str.split(":")
                                new_time = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                                if new_time > current_time:
                                    pbar.update(new_time - current_time)
                            except (IndexError, ValueError):
                                pass
                        break
                    
                    # Use select to avoid blocking (Unix only, fallback for Windows)
                    try:
                        import sys
                        if sys.platform != 'win32':
                            ready, _, _ = select.select([process.stderr], [], [], 1.0)
                            if not ready:
                                # Check for timeout
                                if time.time() - last_activity > read_timeout:
                                    logger.warning("Audio extraction appears stalled, continuing...")
                                    break
                                continue
                        line = process.stderr.readline()
                    except (OSError, ValueError):
                        # Fallback: direct read with timeout check
                        line = process.stderr.readline()
                    
                    if line:
                        last_activity = time.time()
                        if "time=" in line:
                            # Parse time from ffmpeg output (format: time=HH:MM:SS.xx)
                            try:
                                time_str = line.split("time=")[1].split()[0]
                                parts = time_str.split(":")
                                new_time = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                                if new_time > current_time:
                                    pbar.update(new_time - current_time)
                                    current_time = new_time
                            except (IndexError, ValueError):
                                pass
            
            returncode = process.wait(timeout=60)
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, ffmpeg_args)
            
            logger.info("...audio extraction complete: %s", temp_path)
            self._temp_audio_file = temp_path
            return temp_path
        except subprocess.CalledProcessError as e:
            # Cleanup on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            logger.warning("Audio extraction failed, falling back to direct streaming: %s", e)
            raise

    def _cleanup_temp_file(self) -> None:
        """Clean up temporary audio file if exists."""
        if self._temp_audio_file and os.path.exists(self._temp_audio_file):
            try:
                os.unlink(self._temp_audio_file)
                logger.info("Cleaned up temp file: %s", self._temp_audio_file)
            except Exception as e:
                logger.warning("Failed to cleanup temp file: %s", e)
            self._temp_audio_file = None

    def _calculate_segment_positions(self, total_duration: float) -> List[Dict]:
        """Calculate segment start positions distributed across the video.
        
        If skip_intro_outro is enabled, applies start/end margins to skip 
        intro/credits which often lack dialogue.
        
        Args:
            total_duration: Total video duration in seconds.
            
        Returns:
            List of dicts with 'start', 'duration', and 'index' keys.
        """
        segment_duration = self.SEGMENT_DURATION
        num_segments = self.segment_count
        
        # Apply margins only if skip_intro_outro is enabled
        if self.skip_intro_outro:
            start_margin = self.DEFAULT_START_MARGIN
            end_margin = self.DEFAULT_END_MARGIN
        else:
            start_margin = 0
            end_margin = 0
        
        # Calculate effective range after applying margins
        effective_start = start_margin
        effective_end = total_duration - end_margin
        effective_duration = effective_end - effective_start
        
        # Validate margins don't exceed video duration
        if effective_duration <= 0:
            # Margins too large for this video, reduce or disable them
            logger.warning(
                "Start margin (%ds) + end margin (%ds) exceeds video duration (%.1fs), reducing margins",
                start_margin, end_margin, total_duration
            )
            # Try to keep some margin if possible
            if total_duration > segment_duration * 2:
                # Use 10% of duration as margin on each side
                start_margin = int(total_duration * 0.1)
                end_margin = int(total_duration * 0.1)
                effective_start = start_margin
                effective_end = total_duration - end_margin
                effective_duration = effective_end - effective_start
            else:
                # Video too short for margins, use full duration
                start_margin = 0
                end_margin = 0
                effective_start = 0
                effective_end = total_duration
                effective_duration = total_duration
        
        logger.debug(
            "Segment calculation: total=%.1fs, margins=[%d, %d], effective=[%d, %.1f] (%.1fs)",
            total_duration, start_margin, end_margin, effective_start, effective_end, effective_duration
        )
        
        # Check if video is too short for multi-segment sync
        if effective_duration < segment_duration * 2:
            # Video too short, use single segment in the middle
            logger.warning(
                "Effective duration (%.1f) too short for multi-segment sync, using single segment",
                effective_duration
            )
            # Determine actual segment duration (may be shorter than default)
            actual_segment_duration = min(segment_duration, effective_duration)
            if actual_segment_duration < 10:
                # Segment too short for reliable VAD analysis
                logger.warning(
                    "Segment duration (%.1f) too short for reliable analysis, using full video duration",
                    actual_segment_duration
                )
                actual_segment_duration = min(segment_duration, total_duration)
                center_start = max(0, (total_duration - actual_segment_duration) / 2)
            else:
                # Place single segment in the middle of effective range
                center_start = effective_start + max(0, (effective_duration - actual_segment_duration) / 2)
            
            # Ensure we don't exceed bounds
            center_start = max(0, min(center_start, total_duration - actual_segment_duration))
            return [{'start': int(center_start), 'duration': int(actual_segment_duration), 'index': 0}]
        
        # Adjust segment count if effective duration is too short
        min_duration_needed = num_segments * segment_duration
        if effective_duration < min_duration_needed:
            num_segments = max(2, int(effective_duration / segment_duration))
            logger.info(
                "Adjusted segment count from %d to %d for effective duration %.1f",
                self.segment_count, num_segments, effective_duration
            )
        
        # Distribute segments evenly across the effective range
        # Calculate interval between segment start positions
        usable_range = effective_duration - segment_duration
        if num_segments == 1:
            interval = 0
        else:
            interval = usable_range / (num_segments - 1)
        
        segments = []
        for i in range(num_segments):
            # Calculate start position within effective range
            start = effective_start + int(i * interval)
            
            # Boundary check: ensure segment doesn't exceed video end
            if start + segment_duration > total_duration:
                start = max(0, int(total_duration - segment_duration))
                logger.debug("Segment %d adjusted to %d to avoid exceeding video end", i, start)
            
            # Boundary check: ensure start is not negative
            if start < 0:
                start = 0
                logger.debug("Segment %d adjusted to 0 to avoid negative start", i)
            
            segments.append({
                'start': start,
                'duration': segment_duration,
                'index': i
            })
        
        logger.info(
            "Calculated %d segments with margins [%d, %d]: %s", 
            len(segments), start_margin, end_margin,
            [(s['start'], s['start'] + s['duration']) for s in segments]
        )
        return segments

    def _extract_segment_audio(self, url: str, start: int, duration: int) -> Optional[str]:
        """Extract a specific segment of audio from URL to temp file.
        
        Args:
            url: Remote URL or local file path.
            start: Start time in seconds.
            duration: Duration in seconds.
            
        Returns:
            Path to temp audio file, or None if extraction failed.
        """
        fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        
        ffmpeg_args = [
            ffmpeg_bin_path("ffmpeg", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path),
            "-loglevel", "warning",
            "-ss", str(start),
            "-i", url,
            "-t", str(duration),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.frame_rate),
            "-ac", "1",
            "-y", temp_path
        ]
        
        try:
            result = subprocess.run(ffmpeg_args, capture_output=True, timeout=120)
            if result.returncode != 0:
                logger.warning("Failed to extract segment at %ds: %s", start, result.stderr.decode()[:200])
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return None
            return temp_path
        except subprocess.TimeoutExpired:
            logger.warning("Timeout extracting segment at %ds", start)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return None
        except Exception as e:
            logger.warning("Error extracting segment at %ds: %s", start, e)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return None

    def __del__(self):
        """Destructor to ensure temp file cleanup."""
        self._cleanup_temp_file()

    def try_fit_using_embedded_subs(self, fname: str) -> None:
        embedded_subs = []
        embedded_subs_times = []
        if self.ref_stream is None:
            # check first 5; should cover 99% of movies
            streams_to_try: List[str] = list(map("0:s:{}".format, range(5)))
        else:
            streams_to_try = [self.ref_stream]
        for stream in streams_to_try:
            ffmpeg_args = [
                ffmpeg_bin_path(
                    "ffmpeg", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
                )
            ]
            ffmpeg_args.extend(
                [
                    "-loglevel",
                    "fatal",
                    "-nostdin",
                    "-i",
                    fname,
                    "-map",
                    "{}".format(stream),
                    "-f",
                    "srt",
                    "-",
                ]
            )
            process = subprocess.Popen(
                ffmpeg_args, **subprocess_args(include_stdout=True)
            )
            output = io.BytesIO(process.communicate()[0])
            if process.returncode != 0:
                break
            pipe = cast(
                Pipeline,
                make_subtitle_speech_pipeline(start_seconds=self.start_seconds),
            ).fit(output)
            speech_step = pipe.steps[-1][1]
            embedded_subs.append(speech_step)
            embedded_subs_times.append(speech_step.max_time_)
        if len(embedded_subs) == 0:
            if self.ref_stream is None:
                error_msg = "Video file appears to lack subtitle stream"
            else:
                error_msg = "Stream {} not found".format(self.ref_stream)
            raise ValueError(error_msg)
        # use longest set of embedded subs
        subs_to_use = embedded_subs[int(np.argmax(embedded_subs_times))]
        self.video_speech_results_ = subs_to_use.subtitle_speech_results_

    def fit(self, fname: str, *_) -> "VideoSpeechTransformer":
        # For remote URLs with extract_audio_first, extract audio to temp file
        original_fname = fname
        if self.extract_audio_first and is_remote_url(fname):
            try:
                fname = self._extract_audio_to_temp(fname)
            except Exception as e:
                logger.warning("Failed to extract audio, using direct streaming: %s", e)
                fname = original_fname
        
        try:
            return self._fit_impl(fname)
        finally:
            # Clean up temp file after processing
            self._cleanup_temp_file()

    def _fit_impl(self, fname: str) -> "VideoSpeechTransformer":
        """Internal implementation of fit method."""
        if "subs" in self.vad and (
            self.ref_stream is None or self.ref_stream.startswith("0:s:")
        ):
            try:
                logger.info("Checking video for subtitles stream...")
                self.try_fit_using_embedded_subs(fname)
                logger.info("...success!")
                return self
            except Exception as e:
                logger.info(e)
        # Get total duration and apply smart max_duration_seconds adjustment
        effective_max_duration = self.max_duration_seconds
        try:
            total_duration = (
                float(
                    ffmpeg.probe(
                        fname,
                        cmd=ffmpeg_bin_path(
                            "ffprobe",
                            self.gui_mode,
                            ffmpeg_resources_path=self.ffmpeg_path,
                        ),
                    )["format"]["duration"]
                )
                - self.start_seconds
            )
            # Smart adjustment: if max_duration_seconds exceeds video duration, use actual duration
            if self.max_duration_seconds and total_duration:
                if self.max_duration_seconds > total_duration:
                    logger.info(
                        "max_duration_seconds (%d) exceeds video duration (%.1f), using actual duration",
                        self.max_duration_seconds, total_duration
                    )
                    effective_max_duration = int(total_duration)
                else:
                    total_duration = min(total_duration, self.max_duration_seconds)
        except Exception as e:
            logger.warning(e)
            total_duration = None
        if "webrtc" in self.vad:
            detector = _make_webrtcvad_detector(
                self.sample_rate, self.frame_rate, self._non_speech_label
            )
        elif "auditok" in self.vad:
            detector = _make_auditok_detector(
                self.sample_rate, self.frame_rate, self._non_speech_label
            )
        elif "silero" in self.vad:
            detector = _make_silero_detector(
                self.sample_rate, self.frame_rate, self._non_speech_label
            )
        elif "fused" in self.vad:
            # Extract fusion strategy from vad string (e.g., "fused:weighted", "fused:intersection")
            if ":" in self.vad:
                fusion_strategy = self.vad.split(":")[1]
            else:
                fusion_strategy = "weighted"
            detector = _make_fused_detector(
                self.sample_rate, self.frame_rate, self._non_speech_label,
                fusion_strategy=fusion_strategy
            )
        else:
            raise ValueError("unknown vad: %s" % self.vad)
        media_bstring: List[np.ndarray] = []
        ffmpeg_args = [
            ffmpeg_bin_path(
                "ffmpeg", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
            )
        ]
        if self.start_seconds > 0:
            ffmpeg_args.extend(
                [
                    "-ss",
                    str(timedelta(seconds=self.start_seconds)),
                ]
            )
        ffmpeg_args.extend(["-loglevel", "fatal", "-nostdin", "-i", fname])
        # Add duration limit if specified (for faster processing)
        if effective_max_duration and not self._temp_audio_file:
            # Only add -t if not using pre-extracted audio (which already has duration limit)
            ffmpeg_args.extend(["-t", str(effective_max_duration)])
        if self.ref_stream is not None and self.ref_stream.startswith("0:a:"):
            ffmpeg_args.extend(["-map", self.ref_stream])
        ffmpeg_args.extend(
            [
                "-f",
                "s16le",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-af",
                "aresample=async=1",
                "-ar",
                str(self.frame_rate),
                "-",
            ]
        )
        process = subprocess.Popen(ffmpeg_args, **subprocess_args(include_stdout=True))
        bytes_per_frame = 2
        frames_per_window = bytes_per_frame * self.frame_rate // self.sample_rate
        windows_per_buffer = 10000
        simple_progress = 0.0

        redirect_stderr = None
        tqdm_extra_args = {}
        should_print_redirected_stderr = self.gui_mode
        if self.gui_mode:
            try:
                from contextlib import redirect_stderr  # type: ignore

                tqdm_extra_args["file"] = sys.stdout
            except ImportError:
                should_print_redirected_stderr = False
        if redirect_stderr is None:

            @contextmanager
            def redirect_stderr(enter_result=None):
                yield enter_result

        assert redirect_stderr is not None
        pbar_output = io.StringIO()
        with redirect_stderr(pbar_output):
            with tqdm.tqdm(
                total=total_duration, disable=self.vlc_mode, **tqdm_extra_args
            ) as pbar:
                while True:
                    in_bytes = process.stdout.read(
                        frames_per_window * windows_per_buffer
                    )
                    if not in_bytes:
                        break
                    newstuff = len(in_bytes) / float(bytes_per_frame) / self.frame_rate
                    if (
                        total_duration is not None
                        and simple_progress + newstuff > total_duration
                    ):
                        newstuff = total_duration - simple_progress
                    simple_progress += newstuff
                    pbar.update(newstuff)
                    if self.vlc_mode and total_duration is not None:
                        print("%d" % int(simple_progress * 100.0 / total_duration))
                        sys.stdout.flush()
                    if should_print_redirected_stderr:
                        assert self.gui_mode
                        # no need to flush since we pass -u to do unbuffered output for gui mode
                        print(pbar_output.read())
                    if "silero" not in self.vad:
                        in_bytes = np.frombuffer(in_bytes, np.uint8)
                    media_bstring.append(detector(in_bytes))
        process.wait()
        if len(media_bstring) == 0:
            raise ValueError(
                "Unable to detect speech. "
                "Perhaps try specifying a different stream / track, or a different vad."
            )
        self.video_speech_results_ = np.concatenate(media_bstring)
        logger.info("total of speech segments: %s", np.sum(self.video_speech_results_))
        return self

    def transform(self, *_) -> np.ndarray:
        return self.video_speech_results_


_PAIRED_NESTER: Dict[str, str] = {
    "(": ")",
    "{": "}",
    "[": "]",
    # FIXME: False positive sometimes when there are html tags, e.g. <i> Hello? </i>
    # '<': '>',
}


# TODO: need way better metadata detector
def _is_metadata(content: str, is_beginning_or_end: bool) -> bool:
    content = content.strip()
    if len(content) == 0:
        return True
    if (
        content[0] in _PAIRED_NESTER.keys()
        and content[-1] == _PAIRED_NESTER[content[0]]
    ):
        return True
    if is_beginning_or_end:
        if "english" in content.lower():
            return True
        if " - " in content:
            return True
    return False


class SubtitlePreprocessorTransformer(TransformerMixin):
    """Transformer to preprocess subtitles before alignment."""
    
    def __init__(
        self,
        filter_non_dialogue: bool = True,
        merge_short: bool = True,
        min_duration: float = 0.3,
        max_gap: float = 0.3,
        min_keep_ratio: float = 0.3
    ) -> None:
        super(SubtitlePreprocessorTransformer, self).__init__()
        self.filter_non_dialogue = filter_non_dialogue
        self.merge_short = merge_short
        self.min_duration = min_duration
        self.max_gap = max_gap
        self.min_keep_ratio = min_keep_ratio
        self.subs_ = None
    
    def fit(self, subs, *_) -> "SubtitlePreprocessorTransformer":
        # subs can be GenericSubtitlesFile or list of GenericSubtitle
        if hasattr(subs, '__iter__') and not isinstance(subs, (str, bytes)):
            subs_list = list(subs)
        else:
            subs_list = subs
        
        processed = preprocess_subtitles(
            subs_list,
            filter_non_dialogue=self.filter_non_dialogue,
            merge_short=self.merge_short,
            min_duration=self.min_duration,
            max_gap=self.max_gap,
            min_keep_ratio=self.min_keep_ratio
        )
        
        # Preserve GenericSubtitlesFile wrapper if present
        if hasattr(subs, 'clone_props_for_subs'):
            self.subs_ = subs.clone_props_for_subs(processed)
        else:
            self.subs_ = processed
        return self
    
    def transform(self, *_):
        return self.subs_


class SubtitleSpeechTransformer(TransformerMixin, ComputeSpeechFrameBoundariesMixin):
    def __init__(
        self, sample_rate: int, start_seconds: int = 0, framerate_ratio: float = 1.0
    ) -> None:
        super(SubtitleSpeechTransformer, self).__init__()
        self.sample_rate: int = sample_rate
        self.start_seconds: int = start_seconds
        self.framerate_ratio: float = framerate_ratio
        self.subtitle_speech_results_: Optional[np.ndarray] = None
        self.max_time_: Optional[int] = None

    def fit(self, subs: List[GenericSubtitle], *_) -> "SubtitleSpeechTransformer":
        max_time = 0
        for sub in subs:
            max_time = max(max_time, sub.end.total_seconds())
        self.max_time_ = max_time - self.start_seconds
        samples = np.zeros(int(max_time * self.sample_rate) + 2, dtype=float)
        start_frame = float("inf")
        end_frame = 0
        for i, sub in enumerate(subs):
            if _is_metadata(sub.content, i == 0 or i + 1 == len(subs)):
                continue
            start = int(
                round(
                    (sub.start.total_seconds() - self.start_seconds) * self.sample_rate
                )
            )
            start_frame = min(start_frame, start)
            duration = sub.end.total_seconds() - sub.start.total_seconds()
            end = start + int(round(duration * self.sample_rate))
            end_frame = max(end_frame, end)
            samples[start:end] = min(1.0 / self.framerate_ratio, 1.0)
        self.subtitle_speech_results_ = samples
        self.fit_boundaries(self.subtitle_speech_results_)
        return self

    def transform(self, *_) -> np.ndarray:
        assert self.subtitle_speech_results_ is not None
        return self.subtitle_speech_results_


class DeserializeSpeechTransformer(TransformerMixin):
    def __init__(self, non_speech_label: float) -> None:
        super(DeserializeSpeechTransformer, self).__init__()
        self._non_speech_label: float = non_speech_label
        self.deserialized_speech_results_: Optional[np.ndarray] = None

    def fit(self, fname, *_) -> "DeserializeSpeechTransformer":
        speech = np.load(fname)
        if hasattr(speech, "files"):
            if "speech" in speech.files:
                speech = speech["speech"]
            else:
                raise ValueError(
                    'could not find "speech" array in '
                    "serialized file; only contains: %s" % speech.files
                )
        speech[speech < 1.0] = self._non_speech_label
        self.deserialized_speech_results_ = speech
        return self

    def transform(self, *_) -> np.ndarray:
        assert self.deserialized_speech_results_ is not None
        return self.deserialized_speech_results_
