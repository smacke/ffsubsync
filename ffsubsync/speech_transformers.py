# -*- coding: utf-8 -*-
from contextlib import contextmanager
import logging
import io
import subprocess
import sys
from datetime import timedelta

import ffmpeg
import numpy as np
from .sklearn_shim import TransformerMixin
from .sklearn_shim import Pipeline
import tqdm

from .constants import *
from .ffmpeg_utils import ffmpeg_bin_path, subprocess_args
from .subtitle_parser import make_subtitle_parser
from .subtitle_transformers import SubtitleScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_subtitle_speech_pipeline(
        fmt='srt',
        encoding=DEFAULT_ENCODING,
        caching=False,
        max_subtitle_seconds=DEFAULT_MAX_SUBTITLE_SECONDS,
        start_seconds=DEFAULT_START_SECONDS,
        scale_factor=DEFAULT_SCALE_FACTOR,
        parser=None,
        **kwargs
):
    if parser is None:
        parser = make_subtitle_parser(
            fmt,
            encoding=encoding,
            caching=caching,
            max_subtitle_seconds=max_subtitle_seconds,
            start_seconds=start_seconds,
            **kwargs
        )
    assert parser.encoding == encoding
    assert parser.max_subtitle_seconds == max_subtitle_seconds
    assert parser.start_seconds == start_seconds

    def subpipe_maker(framerate_ratio):
        return Pipeline([
            ('parse', parser),
            ('scale', SubtitleScaler(framerate_ratio)),
            ('speech_extract', SubtitleSpeechTransformer(
                sample_rate=SAMPLE_RATE,
                start_seconds=start_seconds,
                framerate_ratio=framerate_ratio,
            ))
        ])
    if scale_factor is None:
        return subpipe_maker
    else:
        return subpipe_maker(scale_factor)


def _make_auditok_detector(sample_rate, frame_rate, non_speech_label):
    try:
        from auditok import \
            BufferAudioSource, ADSFactory, AudioEnergyValidator, StreamTokenizer
    except ImportError as e:
        logger.error("""Error: auditok not installed!
        Consider installing it with `pip install auditok`. Note that auditok
        is GPLv3 licensed, which means that successfully importing it at
        runtime creates a derivative work that is GPLv3 licensed. For personal
        use this is fine, but note that any commercial use that relies on
        auditok must be open source as per the GPLv3!*
        *Not legal advice. Consult with a lawyer.
        """)
        raise e
    bytes_per_frame = 2
    frames_per_window = frame_rate // sample_rate
    validator = AudioEnergyValidator(
        sample_width=bytes_per_frame, energy_threshold=50
    )
    tokenizer = StreamTokenizer(
        validator=validator,
        min_length=0.2 * sample_rate,
        max_length=int(5 * sample_rate),
        max_continuous_silence=0.25 * sample_rate
    )

    def _detect(asegment):
        asource = BufferAudioSource(
            data_buffer=asegment,
            sampling_rate=frame_rate,
            sample_width=bytes_per_frame,
            channels=1
        )
        ads = ADSFactory.ads(audio_source=asource, block_dur=1./sample_rate)
        ads.open()
        tokens = tokenizer.tokenize(ads)
        length = (
            len(asegment)//bytes_per_frame + frames_per_window - 1
        ) // frames_per_window
        media_bstring = np.zeros(length + 1)
        for token in tokens:
            media_bstring[token[1]] = 1.
            media_bstring[token[2] + 1] = non_speech_label - 1.
        return np.clip(np.cumsum(media_bstring)[:-1], 0., 1.)
    return _detect


def _make_webrtcvad_detector(sample_rate, frame_rate, non_speech_label):
    import webrtcvad
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # set non-speech pruning aggressiveness from 0 to 3
    window_duration = 1. / sample_rate  # duration in seconds
    frames_per_window = int(window_duration * frame_rate + 0.5)
    bytes_per_frame = 2

    def _detect(asegment):
        media_bstring = []
        failures = 0
        for start in range(0, len(asegment) // bytes_per_frame,
                           frames_per_window):
            stop = min(start + frames_per_window,
                       len(asegment) // bytes_per_frame)
            try:
                is_speech = vad.is_speech(
                    asegment[start * bytes_per_frame: stop * bytes_per_frame],
                    sample_rate=frame_rate)
            except:
                is_speech = False
                failures += 1
            # webrtcvad has low recall on mode 3, so treat non-speech as "not sure"
            media_bstring.append(1. if is_speech else non_speech_label)
        return np.array(media_bstring)

    return _detect


class ComputeSpeechFrameBoundariesMixin(object):
    def __init__(self):
        self.start_frame_ = None
        self.end_frame_ = None

    @property
    def num_frames(self):
        if self.start_frame_ is None or self.end_frame_ is None:
            return None
        return self.end_frame_ - self.start_frame_

    def fit_boundaries(self, speech_frames):
        nz = np.nonzero(speech_frames > 0.5)[0]
        if len(nz) > 0:
            self.start_frame_ = np.min(nz)
            self.end_frame_ = np.max(nz)
        return self


class VideoSpeechTransformer(TransformerMixin):
    def __init__(
        self, vad, sample_rate, frame_rate, non_speech_label, start_seconds=0,
        ffmpeg_path=None, ref_stream=None, vlc_mode=False, gui_mode=False
    ):
        super(VideoSpeechTransformer, self).__init__()
        self.vad = vad
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self._non_speech_label = non_speech_label
        self.start_seconds = start_seconds
        self.ffmpeg_path = ffmpeg_path
        self.ref_stream = ref_stream
        self.vlc_mode = vlc_mode
        self.gui_mode = gui_mode
        self.video_speech_results_ = None

    def try_fit_using_embedded_subs(self, fname):
        embedded_subs = []
        embedded_subs_times = []
        if self.ref_stream is None:
            # check first 5; should cover 99% of movies
            streams_to_try = map('0:s:{}'.format, range(5))
        else:
            streams_to_try = [self.ref_stream]
        for stream in streams_to_try:
            ffmpeg_args = [ffmpeg_bin_path('ffmpeg', self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path)]
            ffmpeg_args.extend([
                '-loglevel', 'fatal',
                '-nostdin',
                '-i', fname,
                '-map', '{}'.format(stream),
                '-f', 'srt',
                '-'
            ])
            process = subprocess.Popen(ffmpeg_args, **subprocess_args(include_stdout=True))
            output = io.BytesIO(process.communicate()[0])
            if process.returncode != 0:
                break
            pipe = make_subtitle_speech_pipeline(start_seconds=self.start_seconds).fit(output)
            speech_step = pipe.steps[-1][1]
            embedded_subs.append(speech_step)
            embedded_subs_times.append(speech_step.max_time_)
        if len(embedded_subs) == 0:
            if self.ref_stream is None:
                error_msg = 'Video file appears to lack subtitle stream'
            else:
                error_msg = 'Stream {} not found'.format(self.ref_stream)
            raise ValueError(error_msg)
        # use longest set of embedded subs
        subs_to_use = embedded_subs[int(np.argmax(embedded_subs_times))]
        self.video_speech_results_ = subs_to_use.subtitle_speech_results_

    def fit(self, fname, *_):
        if 'subs' in self.vad and (self.ref_stream is None or self.ref_stream.startswith('0:s:')):
            try:
                logger.info('Checking video for subtitles stream...')
                self.try_fit_using_embedded_subs(fname)
                logger.info('...success!')
                return self
            except Exception as e:
                logger.info(e)
        try:
            total_duration = float(ffmpeg.probe(
                fname, cmd=ffmpeg_bin_path('ffprobe', self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path)
            )['format']['duration']) - self.start_seconds
        except Exception as e:
            logger.warning(e)
            total_duration = None
        if 'webrtc' in self.vad:
            detector = _make_webrtcvad_detector(self.sample_rate, self.frame_rate, self._non_speech_label)
        elif 'auditok' in self.vad:
            detector = _make_auditok_detector(self.sample_rate, self.frame_rate, self._non_speech_label)
        else:
            raise ValueError('unknown vad: %s' % self.vad)
        media_bstring = []
        ffmpeg_args = [ffmpeg_bin_path('ffmpeg', self.gui_mode, ffmpeg_resources_path=self.ffmpeg_path)]
        if self.start_seconds > 0:
            ffmpeg_args.extend([
                '-ss', str(timedelta(seconds=self.start_seconds)),
            ])
        ffmpeg_args.extend([
            '-loglevel', 'fatal',
            '-nostdin',
            '-i', fname
        ])
        if self.ref_stream is not None and self.ref_stream.startswith('0:a:'):
            ffmpeg_args.extend(['-map', self.ref_stream])
        ffmpeg_args.extend([
            '-f', 's16le',
            '-ac', '1',
            '-acodec', 'pcm_s16le',
            '-ar', str(self.frame_rate),
            '-'
        ])
        process = subprocess.Popen(ffmpeg_args, **subprocess_args(include_stdout=True))
        bytes_per_frame = 2
        frames_per_window = bytes_per_frame * self.frame_rate // self.sample_rate
        windows_per_buffer = 10000
        simple_progress = 0.

        @contextmanager
        def redirect_stderr(enter_result=None):
            yield enter_result
        tqdm_extra_args = {}
        should_print_redirected_stderr = self.gui_mode
        if self.gui_mode:
            try:
                from contextlib import redirect_stderr
                tqdm_extra_args['file'] = sys.stdout
            except ImportError:
                should_print_redirected_stderr = False
        pbar_output = io.StringIO()
        with redirect_stderr(pbar_output):
            with tqdm.tqdm(total=total_duration, disable=self.vlc_mode, **tqdm_extra_args) as pbar:
                while True:
                    in_bytes = process.stdout.read(frames_per_window * windows_per_buffer)
                    if not in_bytes:
                        break
                    newstuff = len(in_bytes) / float(bytes_per_frame) / self.frame_rate
                    if total_duration is not None and simple_progress + newstuff > total_duration:
                        newstuff = total_duration - simple_progress
                    simple_progress += newstuff
                    pbar.update(newstuff)
                    if self.vlc_mode and total_duration is not None:
                        print("%d" % int(simple_progress * 100. / total_duration))
                        sys.stdout.flush()
                    if should_print_redirected_stderr:
                        assert self.gui_mode
                        # no need to flush since we pass -u to do unbuffered output for gui mode
                        print(pbar_output.read())
                    in_bytes = np.frombuffer(in_bytes, np.uint8)
                    media_bstring.append(detector(in_bytes))
        if len(media_bstring) == 0:
            raise ValueError(
                'Unable to detect speech. Perhaps try specifying a different stream / track, or a different vad.'
            )
        self.video_speech_results_ = np.concatenate(media_bstring)
        return self

    def transform(self, *_):
        return self.video_speech_results_


_PAIRED_NESTER = {
    '(': ')',
    '{': '}',
    '[': ']',
    # FIXME: False positive sometimes when there are html tags, e.g. <i> Hello? </i>
    # '<': '>',
}


# TODO: need way better metadata detector
def _is_metadata(content, is_beginning_or_end):
    content = content.strip()
    if len(content) == 0:
        return True
    if content[0] in _PAIRED_NESTER.keys() and content[-1] == _PAIRED_NESTER[content[0]]:
        return True
    if is_beginning_or_end:
        if 'english' in content.lower():
            return True
        if ' - ' in content:
            return True
    return False


class SubtitleSpeechTransformer(TransformerMixin, ComputeSpeechFrameBoundariesMixin):
    def __init__(self, sample_rate, start_seconds=0, framerate_ratio=1.):
        super(SubtitleSpeechTransformer, self).__init__()
        self.sample_rate = sample_rate
        self.start_seconds = start_seconds
        self.framerate_ratio = framerate_ratio
        self.subtitle_speech_results_ = None
        self.max_time_ = None

    def fit(self, subs, *_):
        max_time = 0
        for sub in subs:
            max_time = max(max_time, sub.end.total_seconds())
        self.max_time_ = max_time - self.start_seconds
        samples = np.zeros(int(max_time * self.sample_rate) + 2, dtype=float)
        start_frame = float('inf')
        end_frame = 0
        for i, sub in enumerate(subs):
            if _is_metadata(sub.content, i == 0 or i + 1 == len(subs)):
                continue
            start = int(round((sub.start.total_seconds() - self.start_seconds) * self.sample_rate))
            start_frame = min(start_frame, start)
            duration = sub.end.total_seconds() - sub.start.total_seconds()
            end = start + int(round(duration * self.sample_rate))
            end_frame = max(end_frame, end)
            samples[start:end] = min(1. / self.framerate_ratio, 1.)
        self.subtitle_speech_results_ = samples
        self.fit_boundaries(self.subtitle_speech_results_)
        return self

    def transform(self, *_):
        return self.subtitle_speech_results_


class DeserializeSpeechTransformer(TransformerMixin):
    def __init__(self, non_speech_label):
        super(DeserializeSpeechTransformer, self).__init__()
        self._non_speech_label = non_speech_label
        self.deserialized_speech_results_ = None

    def fit(self, fname, *_):
        speech = np.load(fname)
        if hasattr(speech, 'files'):
            if 'speech' in speech.files:
                speech = speech['speech']
            else:
                raise ValueError('could not find "speech" array in '
                                 'serialized file; only contains: %s' % speech.files)
        speech[speech < 1.] = self._non_speech_label
        self.deserialized_speech_results_ = speech
        return self

    def transform(self, *_):
        return self.deserialized_speech_results_
