# -*- coding: utf-8 -*-
import os
from concurrent.futures import as_completed, ThreadPoolExecutor
from contextlib import contextmanager
import logging
import io
import re
import subprocess
import sys
import tempfile
from datetime import timedelta
from typing import cast, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

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


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


class ProgressInfo(NamedTuple):
    """Progress emitted to a ``progress_handler`` during speech extraction.

    ``processed_seconds`` is the amount of reference audio decoded so far and
    ``total_seconds`` is the reference's total duration (``None`` when ffprobe
    could not determine it). Use :attr:`fraction` for a 0.0-1.0 ratio.
    """

    processed_seconds: float
    total_seconds: Optional[float]

    @property
    def fraction(self) -> Optional[float]:
        if not self.total_seconds:
            return None
        return min(1.0, self.processed_seconds / self.total_seconds)


def make_subtitle_speech_pipeline(
    fmt: str = "srt",
    encoding: str = DEFAULT_ENCODING,
    caching: bool = False,
    max_subtitle_seconds: int = DEFAULT_MAX_SUBTITLE_SECONDS,
    start_seconds: int = DEFAULT_START_SECONDS,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    parser=None,
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
        return Pipeline(
            [
                ("parse", parser),
                ("scale", SubtitleScaler(framerate_ratio)),
                (
                    "speech_extract",
                    SubtitleSpeechTransformer(
                        sample_rate=SAMPLE_RATE,
                        start_seconds=start_seconds,
                        framerate_ratio=framerate_ratio,
                    ),
                ),
            ]
        )

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
    try:
        import torch
    except ImportError as e:
        logger.error(
            "Error: the silero VAD requires PyTorch, which is not installed!\n"
            "        Install it with `pip install torch` (see https://pytorch.org\n"
            "        for platform-specific instructions). torch is an optional\n"
            "        dependency and is not installed with ffsubsync by default."
        )
        raise e

    window_duration = 1.0 / sample_rate  # duration in seconds
    frames_per_window = int(window_duration * frame_rate + 0.5)
    bytes_per_frame = 1

    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )

    exception_logged = False

    def _detect(asegment) -> np.ndarray:
        asegment = np.frombuffer(asegment, np.int16).astype(np.float32) / (1 << 15)
        asegment = torch.FloatTensor(asegment)
        media_bstring = []
        failures = 0
        for start in range(0, len(asegment) // bytes_per_frame, frames_per_window):
            stop = min(start + frames_per_window, len(asegment))
            try:
                speech_prob = model(
                    asegment[start * bytes_per_frame : stop * bytes_per_frame],
                    frame_rate,
                ).item()
            except Exception:
                nonlocal exception_logged
                if not exception_logged:
                    exception_logged = True
                    logger.exception("exception occurred during speech detection")
                speech_prob = 0.0
                failures += 1
            media_bstring.append(1.0 - (1.0 - speech_prob) * (1.0 - non_speech_label))
        return np.array(media_bstring)

    return _detect


# bitmap (image-based) subtitle codecs cannot be muxed to SRT; mapping one
# into an SRT extraction aborts the whole ffmpeg invocation, so they are
# skipped when enumerating embedded subtitle streams
_BITMAP_SUBTITLE_CODECS: frozenset = frozenset(
    {
        "hdmv_pgs_subtitle",
        "dvd_subtitle",
        "dvb_subtitle",
        "dvb_teletext",
        "xsub",
    }
)


_FUSION_STRATEGIES: Tuple[str, ...] = ("weighted", "intersection", "union")


def _make_fused_detector(
    sample_rate: int,
    frame_rate: int,
    non_speech_label: float,
    fusion_strategy: str = "weighted",
) -> Callable[[bytes], np.ndarray]:
    """Combine the webrtc and silero VADs into a single detector.

    Requires the optional silero dependency (torch); a clear error is raised if
    it is missing. ``fusion_strategy`` controls how the two are combined:
    ``intersection`` (speech only where both agree -- conservative),
    ``union`` (speech where either fires -- aggressive), or ``weighted``
    (default; ``0.6 * silero + 0.4 * webrtc``).
    """
    if fusion_strategy not in _FUSION_STRATEGIES:
        raise ValueError(
            "unknown fused VAD strategy %r; choose one of %s"
            % (fusion_strategy, ", ".join(_FUSION_STRATEGIES))
        )
    webrtc_detector = _make_webrtcvad_detector(
        sample_rate, frame_rate, non_speech_label
    )
    silero_detector = _make_silero_detector(
        sample_rate, frame_rate, non_speech_label
    )

    def _detect(asegment) -> np.ndarray:
        webrtc_result = webrtc_detector(asegment)
        silero_result = silero_detector(asegment)
        # the two detectors can disagree by a frame at the tail; clip to common length
        min_len = min(len(webrtc_result), len(silero_result))
        webrtc_result = webrtc_result[:min_len]
        silero_result = silero_result[:min_len]
        if fusion_strategy == "intersection":
            return np.minimum(webrtc_result, silero_result)
        elif fusion_strategy == "union":
            return np.maximum(webrtc_result, silero_result)
        else:  # weighted
            return 0.6 * silero_result + 0.4 * webrtc_result

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
        max_duration_seconds: Optional[float] = None,
        extract_audio_first: bool = False,
        progress_handler: Optional[Callable[["ProgressInfo"], None]] = None,
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
        self.max_duration_seconds: Optional[float] = max_duration_seconds
        self.extract_audio_first: bool = extract_audio_first
        self.progress_handler: Optional[
            Callable[["ProgressInfo"], None]
        ] = progress_handler
        self.video_speech_results_: Optional[np.ndarray] = None

    def _probe_embedded_subtitle_streams(self, fname: str) -> Optional[List[str]]:
        """Enumerate text-based subtitle streams in ``fname`` via ffprobe.

        Returns a list of ffmpeg ``-map`` specifiers (e.g. ``["0:2", "0:3"]``)
        so that every subtitle stream can be extracted in a single ffmpeg pass.
        Bitmap subtitle codecs (PGS, VobSub, DVB, ...) are skipped because they
        cannot be muxed to SRT and would otherwise abort the whole extraction.

        Returns ``None`` when ffprobe is unavailable or fails, signaling the
        caller to fall back to extracting streams one at a time.
        """
        ffprobe_args = [
            ffmpeg_bin_path(
                "ffprobe", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
            ),
            "-loglevel",
            "fatal",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=index,codec_name",
            "-of",
            "csv=p=0",
            fname,
        ]
        try:
            process = subprocess.Popen(
                ffprobe_args, **subprocess_args(include_stdout=True)
            )
            output = process.communicate()[0]
        except OSError as e:
            logger.warning("ffprobe unavailable while enumerating subtitles: %s", e)
            return None
        if process.returncode != 0:
            return None
        streams: List[str] = []
        for line in output.decode("utf-8", errors="replace").splitlines():
            parts = line.strip().split(",")
            if not parts or not parts[0].isdigit():
                continue
            index = parts[0]
            codec_name = parts[1].strip().lower() if len(parts) > 1 else ""
            if codec_name in _BITMAP_SUBTITLE_CODECS:
                continue
            streams.append("0:{}".format(index))
        return streams or None

    def _extract_embedded_subs_single_pass(
        self, fname: str, streams: List[str]
    ) -> Optional[List[io.BytesIO]]:
        """Extract several subtitle streams in one ffmpeg invocation.

        ffmpeg can only send a single output to stdout, so each stream is
        written to a temporary file (in the system temp dir -- never next to
        the source media), read back into memory, and then deleted along with
        the temp dir. Returns one in-memory buffer per stream that produced
        data, or ``None`` if the ffmpeg invocation failed wholesale (the caller
        then falls back to per-stream extraction).
        """
        with tempfile.TemporaryDirectory(prefix="ffsubsync_subs_") as tmpdir:
            ffmpeg_args = [
                ffmpeg_bin_path(
                    "ffmpeg", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
                ),
                "-loglevel",
                "fatal",
                "-nostdin",
                "-i",
                fname,
            ]
            out_paths: List[str] = []
            for i, stream in enumerate(streams):
                out_path = os.path.join(tmpdir, "embedded.{}.srt".format(i))
                out_paths.append(out_path)
                ffmpeg_args.extend(
                    ["-map", "{}".format(stream), "-f", "srt", out_path]
                )
            process = subprocess.Popen(
                ffmpeg_args, **subprocess_args(include_stdout=True)
            )
            process.communicate()
            if process.returncode != 0:
                return None
            buffers: List[io.BytesIO] = []
            for out_path in out_paths:
                if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                    continue
                with open(out_path, "rb") as f:
                    buffers.append(io.BytesIO(f.read()))
            return buffers

    def _extract_embedded_subs_per_stream(
        self, fname: str, streams: List[str]
    ) -> List[io.BytesIO]:
        """Extract subtitle streams one ffmpeg invocation at a time (to stdout).

        This preserves the original extraction behavior and is used as a
        fallback whenever single-pass extraction is unavailable (no ffprobe) or
        fails. Stops at the first stream ffmpeg cannot extract.
        """
        buffers: List[io.BytesIO] = []
        for stream in streams:
            ffmpeg_args = [
                ffmpeg_bin_path(
                    "ffmpeg", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
                ),
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
            process = subprocess.Popen(
                ffmpeg_args, **subprocess_args(include_stdout=True)
            )
            output = process.communicate()[0]
            if process.returncode != 0:
                break
            buffers.append(io.BytesIO(output))
        return buffers

    def try_fit_using_embedded_subs(self, fname: str) -> None:
        if self.ref_stream is not None:
            # a specific stream was requested; extract just that one
            subtitle_buffers = self._extract_embedded_subs_per_stream(
                fname, [self.ref_stream]
            )
        else:
            # enumerate the subtitle streams so they can all be extracted in a
            # single ffmpeg pass (~5x faster than one invocation per stream)
            streams = self._probe_embedded_subtitle_streams(fname)
            if streams:
                subtitle_buffers = self._extract_embedded_subs_single_pass(
                    fname, streams
                )
                if subtitle_buffers is None:
                    # single pass failed; degrade to per-stream over the same
                    # (known-present) streams
                    subtitle_buffers = self._extract_embedded_subs_per_stream(
                        fname, streams
                    )
            else:
                # ffprobe unavailable/failed: fall back to probing the first 5
                # streams individually (should cover 99% of movies)
                subtitle_buffers = self._extract_embedded_subs_per_stream(
                    fname, list(map("0:s:{}".format, range(5)))
                )
        embedded_subs = []
        embedded_subs_times = []
        for buffer in subtitle_buffers:
            pipe = cast(
                Pipeline,
                make_subtitle_speech_pipeline(start_seconds=self.start_seconds),
            ).fit(buffer)
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

    def _build_ffmpeg_args(self, fname: str) -> List[str]:
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
        if self.max_duration_seconds is not None:
            # input-side -t: stop reading (and, for remote URLs, downloading)
            # after this many seconds past the seek point
            ffmpeg_args.extend(
                ["-t", str(timedelta(seconds=self.max_duration_seconds))]
            )
        ffmpeg_args.extend(["-loglevel", "fatal", "-nostdin", "-i", fname])
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
        return ffmpeg_args

    def _extract_audio_to_temp(self, url: str) -> Optional[str]:
        """Copy the reference's audio to a local temp file (no re-encode).

        Returns the temp path, or None if extraction fails (the caller then
        falls back to streaming the URL directly). Uses a Matroska (.mka)
        container, which accepts essentially any audio codec, so ``-acodec
        copy`` is safe regardless of the source's codec.
        """
        fd, temp_path = tempfile.mkstemp(suffix=".mka")
        os.close(fd)
        ffmpeg_args = [
            ffmpeg_bin_path(
                "ffmpeg", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
            ),
            "-loglevel",
            "fatal",
            "-nostdin",
            "-y",
            "-i",
            url,
            "-vn",
            "-acodec",
            "copy",
        ]
        if self.max_duration_seconds is not None:
            # extract from t=0 up to start+max so the main pass can still seek
            # --start-seconds accurately within the local file
            limit = self.start_seconds + self.max_duration_seconds
            ffmpeg_args.extend(["-t", str(timedelta(seconds=limit))])
        ffmpeg_args.append(temp_path)
        logger.info("extracting audio from remote reference to %s...", temp_path)
        retcode = subprocess.call(ffmpeg_args, **subprocess_args(include_stdout=False))
        if retcode != 0 or not os.path.getsize(temp_path):
            logger.warning(
                "audio extraction failed (ffmpeg returned %d); "
                "falling back to streaming the reference directly",
                retcode,
            )
            try:
                os.remove(temp_path)
            except OSError:
                pass
            return None
        return temp_path

    def fit(self, fname: str, *_) -> "VideoSpeechTransformer":
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
        temp_audio = None
        if self.extract_audio_first and is_remote_url(fname):
            temp_audio = self._extract_audio_to_temp(fname)
            if temp_audio is not None:
                fname = temp_audio
        try:
            self._fit_using_audio(fname)
        finally:
            if temp_audio is not None and os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except OSError:
                    logger.warning("failed to remove temp audio file %s", temp_audio)
        return self

    def _fit_using_audio(self, fname: str) -> None:
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
        except Exception as e:
            logger.warning(e)
            total_duration = None
        if self.max_duration_seconds is not None and total_duration is not None:
            total_duration = min(total_duration, self.max_duration_seconds)
        if "fused" in self.vad:
            # e.g. "fused" or "fused:intersection"; default strategy is weighted
            fusion_strategy = (
                self.vad.split(":", 1)[1] if ":" in self.vad else "weighted"
            )
            detector = _make_fused_detector(
                self.sample_rate,
                self.frame_rate,
                self._non_speech_label,
                fusion_strategy,
            )
        elif "webrtc" in self.vad:
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
        else:
            raise ValueError("unknown vad: %s" % self.vad)
        media_bstring: List[np.ndarray] = []
        ffmpeg_args = self._build_ffmpeg_args(fname)
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
                    if self.progress_handler is not None:
                        try:
                            self.progress_handler(
                                ProgressInfo(
                                    processed_seconds=simple_progress,
                                    total_seconds=total_duration,
                                )
                            )
                        except Exception as e:
                            # a host-supplied callback must never break syncing
                            logger.warning("progress_handler raised: %s", e)
                    if self.vlc_mode and total_duration is not None:
                        print("%d" % int(simple_progress * 100.0 / total_duration))
                        sys.stdout.flush()
                    if should_print_redirected_stderr:
                        assert self.gui_mode
                        # no need to flush since we pass -u to do unbuffered output for gui mode
                        print(pbar_output.read())
                    # silero (and fused, which wraps it) needs the raw s16le
                    # bytes for its own int16 reinterpretation
                    if "silero" not in self.vad and "fused" not in self.vad:
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

    def transform(self, *_) -> np.ndarray:
        return self.video_speech_results_


class MultiSegmentVideoSpeechTransformer(TransformerMixin):
    """Produce a reference speech signal by sampling only a few short segments.

    Instead of extracting and running VAD over the entire reference, this samples
    ``segment_count`` short windows spread across it, runs VAD on each, and places
    the results into a full-length array that is zero everywhere else. That sparse
    signal can be fed straight into the normal alignment path: because it preserves
    each segment's true position on the reference timeline, the existing
    framerate-ratio search and offset cross-correlation recover the same answer
    they would from the full signal -- but only the sampled audio has to be
    extracted (and, for remote URLs, downloaded). Useful for long / remote
    references; for typical local files the plain VideoSpeechTransformer is fine.
    """

    # margins skipped at the very start/end when skip_intro_outro is set, since
    # intros/credits often lack dialogue
    START_MARGIN_SECONDS: int = 30
    END_MARGIN_SECONDS: int = 60

    def __init__(
        self,
        vad: str,
        sample_rate: int,
        frame_rate: int,
        non_speech_label: float,
        segment_count: int = 8,
        segment_duration: int = 60,
        skip_intro_outro: bool = False,
        parallel_workers: int = 4,
        ffmpeg_path: Optional[str] = None,
        ref_stream: Optional[str] = None,
        vlc_mode: bool = False,
        gui_mode: bool = False,
    ) -> None:
        super(MultiSegmentVideoSpeechTransformer, self).__init__()
        # sampling is audio-only, so drop any "subs_then_" prefix (embedded-subtitle
        # extraction ignores the per-segment time window)
        self.vad: str = vad.split("subs_then_")[-1]
        self.sample_rate: int = sample_rate
        self.frame_rate: int = frame_rate
        self._non_speech_label: float = non_speech_label
        self.segment_count: int = segment_count
        self.segment_duration: int = segment_duration
        self.skip_intro_outro: bool = skip_intro_outro
        self.parallel_workers: int = parallel_workers
        self.ffmpeg_path: Optional[str] = ffmpeg_path
        self.ref_stream: Optional[str] = ref_stream
        self.vlc_mode: bool = vlc_mode
        self.gui_mode: bool = gui_mode
        self.video_speech_results_: Optional[np.ndarray] = None

    def _segment_starts(self, total_duration: float) -> List[int]:
        """Evenly-spaced segment start times (seconds) across the reference."""
        duration = self.segment_duration
        if total_duration <= duration:
            return [0]
        start_margin = self.START_MARGIN_SECONDS if self.skip_intro_outro else 0
        end_margin = self.END_MARGIN_SECONDS if self.skip_intro_outro else 0
        lo = float(start_margin)
        hi = total_duration - end_margin
        if hi - lo < duration:  # margins too large for this reference; ignore them
            lo, hi = 0.0, total_duration
        usable = hi - lo - duration
        n = max(1, self.segment_count)
        if usable <= 0 or n == 1:
            return [int(max(0.0, min(lo, total_duration - duration)))]
        step = usable / (n - 1)
        starts = [int(round(lo + i * step)) for i in range(n)]
        # clamp into range and drop duplicates that clamping may create
        starts = [max(0, min(s, int(total_duration) - duration)) for s in starts]
        return sorted(set(starts))

    def _extract_segment_speech(self, fname: str, start: int) -> Tuple[int, np.ndarray]:
        """Run VAD over a single window, returning (start_seconds, speech array)."""
        segment = VideoSpeechTransformer(
            vad=self.vad,
            sample_rate=self.sample_rate,
            frame_rate=self.frame_rate,
            non_speech_label=self._non_speech_label,
            start_seconds=start,
            ffmpeg_path=self.ffmpeg_path,
            ref_stream=self.ref_stream,
            vlc_mode=self.vlc_mode,
            gui_mode=self.gui_mode,
            max_duration_seconds=self.segment_duration,
        )
        segment.fit(fname)
        return start, segment.transform()

    def fit(self, fname: str, *_) -> "MultiSegmentVideoSpeechTransformer":
        try:
            total_duration = float(
                ffmpeg.probe(
                    fname,
                    cmd=ffmpeg_bin_path(
                        "ffprobe", self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path
                    ),
                )["format"]["duration"]
            )
        except Exception as e:
            raise ValueError(
                "multi-segment sync needs the reference duration, but probing "
                "'%s' failed: %s" % (fname, e)
            )
        starts = self._segment_starts(total_duration)
        logger.info(
            "multi-segment sync: sampling %d segment(s) of up to %ds at %s",
            len(starts),
            self.segment_duration,
            [int(s) for s in starts],
        )
        sparse = np.zeros(int(total_duration * self.sample_rate) + 2, dtype=float)
        workers = max(1, min(self.parallel_workers, len(starts)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_start = {
                executor.submit(self._extract_segment_speech, fname, start): start
                for start in starts
            }
            for future in as_completed(future_to_start):
                start = future_to_start[future]
                try:
                    start, seg_speech = future.result()
                except Exception as e:
                    # a single flaky segment shouldn't sink the whole sync; the
                    # remaining segments still localize the offset
                    logger.warning("failed to extract segment at %ds: %s", start, e)
                    continue
                begin = int(start * self.sample_rate)
                end = min(begin + len(seg_speech), len(sparse))
                if end > begin:
                    sparse[begin:end] = seg_speech[: end - begin]
        if not np.any(sparse > 0):
            raise ValueError(
                "Unable to detect speech in any sampled segment. "
                "Perhaps try specifying a different stream / track, or a different vad."
            )
        self.video_speech_results_ = sparse
        logger.info(
            "total of speech segments: %s", np.sum(self.video_speech_results_)
        )
        return self

    def transform(self, *_) -> np.ndarray:
        return self.video_speech_results_


_PAIRED_NESTER: Dict[str, str] = {
    "(": ")",
    "{": "}",
    "[": "]",
    "（": "）",  # full-width / CJK brackets, common in non-English subtitles
    "【": "】",
    "「": "」",
}

# Markup tags (e.g. <i>, </i>, <font ...>) carry no speech. Stripping them
# before classifying a line lets us recognize a wrapped cue like "<i>[music]</i>"
# as non-dialogue while still treating "<i>Hello?</i>" as dialogue -- which is
# why '<' is intentionally not a paired nester above.
_MARKUP_TAG: "re.Pattern[str]" = re.compile(r"<[^>]+>")

# Symbols that, on their own, denote a musical / non-speech cue.
_NON_DIALOGUE_SYMBOLS: frozenset = frozenset("♪♫♬♩🎵🎶")


# TODO: need way better metadata detector
def _is_metadata(content: str, is_beginning_or_end: bool) -> bool:
    content = _MARKUP_TAG.sub("", content).strip()
    if len(content) == 0:
        return True
    if (
        content[0] in _PAIRED_NESTER.keys()
        and content[-1] == _PAIRED_NESTER[content[0]]
    ):
        return True
    # lines consisting only of music notes / sound symbols are cues, not speech
    if all(ch.isspace() or ch in _NON_DIALOGUE_SYMBOLS for ch in content):
        return True
    if is_beginning_or_end:
        if "english" in content.lower():
            return True
        if " - " in content:
            return True
    return False


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


def find_pgs_stream(
    fname: str,
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
) -> Optional[str]:
    """Return the ffmpeg stream specifier for the first PGS subtitle track in *fname*.

    Uses ``ffprobe`` to inspect the file.  Returns a string like ``"0:s:0"`` on
    success, or ``None`` if the file has no ``hdmv_pgs_subtitle`` streams.
    """
    try:
        probe = ffmpeg.probe(
            fname,
            cmd=ffmpeg_bin_path("ffprobe", gui_mode, ffmpeg_resources_path=ffmpeg_path),
        )
    except Exception as e:
        logger.warning("ffprobe failed while searching for PGS streams: %s", e)
        return None

    sub_index = 0
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "subtitle":
            if stream.get("codec_name") == "hdmv_pgs_subtitle":
                specifier = "0:s:{}".format(sub_index)
                logger.info(
                    "auto-detected PGS stream: %s (ffmpeg stream index %s)",
                    specifier,
                    stream.get("index"),
                )
                return specifier
            sub_index += 1

    return None


def _get_pgs_timings_via_ffprobe(
    fname: str,
    stream: str,
    ffmpeg_path: Optional[str] = None,
    gui_mode: bool = False,
) -> Optional[List[Tuple[float, float]]]:
    """Read PGS timings from container metadata using ffprobe.

    MKV stores per-packet PTS and duration for subtitle streams, so we can
    get start/end timestamps without extracting or parsing the raw SUP binary.
    Show events are large packets with a numeric ``duration_time``; clear events
    are tiny (~30-byte) packets with ``duration_time=N/A``.

    Returns a list of ``(start_seconds, end_seconds)`` tuples, or ``None`` if
    ffprobe fails or returns no usable durations.
    """
    ffprobe_cmd = ffmpeg_bin_path(
        "ffprobe", gui_mode, ffmpeg_resources_path=ffmpeg_path
    )
    # ffprobe -select_streams does not accept the "0:" input-index prefix;
    # strip it so "0:s:0" → "s:0" and "0:3" → "3".
    probe_stream = stream[2:] if stream.startswith("0:") else stream
    try:
        probe_data = ffmpeg.probe(
            fname,
            cmd=ffprobe_cmd,
            show_packets=None,
            select_streams=probe_stream,
            show_entries="packet=pts_time,duration_time,size",
        )
    except Exception:
        return None

    results: List[Tuple[float, float]] = []
    for packet in probe_data.get("packets", []):
        pts_time_str = packet.get("pts_time")
        duration_time_str = packet.get("duration_time")
        size_str = packet.get("size")
        if pts_time_str is None or duration_time_str is None or size_str is None:
            continue
        if duration_time_str == "N/A":
            continue
        try:
            pts_time = float(pts_time_str)
            duration_time = float(duration_time_str)
            size = int(size_str)
        except ValueError:
            continue
        if size > 50:  # skip clear events (~30 bytes)
            results.append((pts_time, pts_time + duration_time))

    if not results:
        return None
    return results


class PGSSpeechTransformer(TransformerMixin, ComputeSpeechFrameBoundariesMixin):
    """Use PGS (Presentation Graphic Stream) subtitle timings as a sync reference.

    PGS subtitles are bitmap-based (e.g. Blu-ray / HDMV) and cannot be converted
    to text by ffmpeg, so they can't be fed through the normal subtitle pipeline.
    However, when muxed into an MKV the container still stores a presentation
    timestamp (``pts_time``) and ``duration_time`` for every subtitle packet, so
    we can recover *when* each caption is on screen without decoding the bitmaps
    or parsing the raw SUP/PCS binary at all.

    This transformer reads those per-packet timings via ``ffprobe`` (see
    :func:`_get_pgs_timings_via_ffprobe`), filtering out the tiny "clear" packets
    that carry no image, and builds the same kind of sparse binary speech signal
    that :class:`SubtitleSpeechTransformer` produces for text subtitles: 1.0 while
    a caption is displayed, 0.0 otherwise. That signal can then be aligned against
    the input subtitle file by the normal ffsubsync pipeline.

    The reference stream may be given explicitly via ``ref_stream`` (with or
    without a leading ``0:``), or left as ``None`` to auto-detect the first
    ``hdmv_pgs_subtitle`` track in the file.
    """

    # PGS is already in the MKV timebase so its duration cannot be compared
    # against the SRT to infer a framerate ratio.  Returning None here prevents
    # the duration-based framerate inference in try_sync from running.
    @property
    def num_frames(self) -> None:
        return None

    def __init__(
        self,
        sample_rate: int,
        start_seconds: int = 0,
        ffmpeg_path: Optional[str] = None,
        ref_stream: Optional[str] = None,
        gui_mode: bool = False,
    ) -> None:
        super(PGSSpeechTransformer, self).__init__()
        self.sample_rate: int = sample_rate
        self.start_seconds: int = start_seconds
        self.ffmpeg_path: Optional[str] = ffmpeg_path
        self.ref_stream: Optional[str] = ref_stream
        self.gui_mode: bool = gui_mode
        self.pgs_speech_results_: Optional[np.ndarray] = None

    def fit(self, fname: str, *_) -> "PGSSpeechTransformer":
        if self.ref_stream is None:
            stream = find_pgs_stream(fname, self.ffmpeg_path, self.gui_mode)
            if stream is None:
                raise ValueError(
                    "No hdmv_pgs_subtitle stream found in {}. "
                    "Specify one explicitly with --pgs-ref-stream.".format(fname)
                )
        else:
            stream = self.ref_stream
            if not stream.startswith("0:"):
                stream = "0:" + stream

        logger.info("reading PGS timings for stream %s from %s...", stream, fname)
        timings = _get_pgs_timings_via_ffprobe(
            fname, stream, self.ffmpeg_path, self.gui_mode
        )
        if timings is None:
            raise ValueError(
                "Failed to get PGS timings via ffprobe for stream {} from {}. "
                "Make sure the stream exists and is an hdmv_pgs_subtitle track "
                "(check with: ffprobe -show_streams {}).".format(stream, fname, fname)
            )

        if not timings:
            raise ValueError(
                "No subtitle timings found in PGS stream {}.".format(stream)
            )

        logger.info("found %d PGS subtitle segments", len(timings))
        for i, (s, e) in enumerate(timings[:8]):
            logger.debug(
                "  PGS[%d]: %s --> %s (%.3fs)",
                i,
                str(timedelta(seconds=s)),
                str(timedelta(seconds=e)),
                e - s,
            )

        max_time = max(end for _, end in timings)
        num_samples = int(max_time * self.sample_rate) + 2
        samples = np.zeros(num_samples, dtype=float)

        for start, end in timings:
            start_sample = int(round((start - self.start_seconds) * self.sample_rate))
            end_sample = int(round((end - self.start_seconds) * self.sample_rate))
            start_sample = max(start_sample, 0)
            end_sample = min(end_sample, num_samples)
            if start_sample < end_sample:
                samples[start_sample:end_sample] = 1.0

        self.pgs_speech_results_ = samples
        self.fit_boundaries(self.pgs_speech_results_)
        logger.info(
            "total PGS subtitle frames: %d", int(np.sum(self.pgs_speech_results_))
        )
        return self

    def transform(self, *_) -> np.ndarray:
        assert self.pgs_speech_results_ is not None
        return self.pgs_speech_results_
