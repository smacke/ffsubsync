# -*- coding: utf-8 -*-
from contextlib import contextmanager
import logging
import io
import subprocess
import sys
from datetime import timedelta

import ffmpeg
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import tqdm
import webrtcvad

from .constants import *
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
            start_seconds=start_seconds
        )
    assert parser.encoding == encoding
    assert parser.max_subtitle_seconds == max_subtitle_seconds
    assert parser.start_seconds == start_seconds
    return Pipeline([
        ('parse', parser),
        ('scale', SubtitleScaler(scale_factor)),
        ('speech_extract', SubtitleSpeechTransformer(
            sample_rate=SAMPLE_RATE,
            start_seconds=start_seconds,
            framerate_ratio=scale_factor,
        ))
    ])


def _make_auditok_detector(sample_rate, frame_rate):
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
        sample_width=bytes_per_frame, energy_threshold=50)
    tokenizer = StreamTokenizer(
        validator=validator, min_length=0.2*sample_rate,
        max_length=int(5*sample_rate),
        max_continuous_silence=0.25*sample_rate)

    def _detect(asegment):
        asource = BufferAudioSource(data_buffer=asegment,
                                    sampling_rate=frame_rate,
                                    sample_width=bytes_per_frame,
                                    channels=1)
        ads = ADSFactory.ads(audio_source=asource, block_dur=1./sample_rate)
        ads.open()
        tokens = tokenizer.tokenize(ads)
        length = (len(asegment)//bytes_per_frame
                  + frames_per_window - 1)//frames_per_window
        media_bstring = np.zeros(length+1, dtype=int)
        for token in tokens:
            media_bstring[token[1]] += 1
            media_bstring[token[2]+1] -= 1
        return (np.cumsum(media_bstring)[:-1] > 0).astype(float)
    return _detect


def _make_webrtcvad_detector(sample_rate, frame_rate):
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
            media_bstring.append(1. if is_speech else 0.5)
        return np.array(media_bstring)

    return _detect


class VideoSpeechTransformer(TransformerMixin):
    def __init__(self, vad, sample_rate, frame_rate, start_seconds=0, vlc_mode=False, gui_mode=False):
        self.vad = vad
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.start_seconds = start_seconds
        self.vlc_mode = vlc_mode
        self.gui_mode = gui_mode
        self.video_speech_results_ = None

    def try_fit_using_embedded_subs(self, fname):
        embedded_subs = []
        embedded_subs_times = []
        # check first 5; should cover 99% of movies
        for stream in range(5):
            ffmpeg_args = ['ffmpeg']
            ffmpeg_args.extend([
                '-loglevel', 'fatal',
                '-nostdin',
                '-i', fname,
                '-map', '0:s:{}'.format(stream),
                '-f', 'srt',
                '-'
            ])
            process = subprocess.Popen(ffmpeg_args, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output = io.BytesIO(process.communicate()[0])
            if process.returncode != 0:
                break
            pipe = make_subtitle_speech_pipeline(start_seconds=self.start_seconds).fit(output)
            speech_step = pipe.steps[-1][1]
            embedded_subs.append(speech_step.subtitle_speech_results_)
            embedded_subs_times.append(speech_step.max_time_)
        if len(embedded_subs) == 0:
            raise ValueError('Video file appears to lack subtitle stream')
        # use longest set of embedded subs
        self.video_speech_results_ = embedded_subs[int(np.argmax(embedded_subs_times))]

    def fit(self, fname, *_):
        if 'subs' in self.vad:
            try:
                logger.info('Checking video for subtitles stream...')
                self.try_fit_using_embedded_subs(fname)
                logger.info('...success!')
                return self
            except Exception as e:
                logger.info(e)
        try:
            total_duration = float(ffmpeg.probe(fname)['format']['duration']) - self.start_seconds
        except Exception as e:
            logger.warning(e)
            total_duration = None
        if 'webrtc' in self.vad:
            detector = _make_webrtcvad_detector(self.sample_rate, self.frame_rate)
        elif 'auditok' in self.vad:
            detector = _make_auditok_detector(self.sample_rate, self.frame_rate)
        else:
            raise ValueError('unknown vad: %s' % self.vad)
        media_bstring = []
        ffmpeg_args = ['ffmpeg']
        if self.start_seconds > 0:
            ffmpeg_args.extend([
                '-ss', str(timedelta(seconds=self.start_seconds)),
            ])
        ffmpeg_args.extend([
            '-loglevel', 'fatal',
            '-nostdin',
            '-i', fname,
            '-f', 's16le',
            '-ac', '1',
            '-acodec', 'pcm_s16le',
            '-ar', str(self.frame_rate),
            '-'
        ])
        process = subprocess.Popen(ffmpeg_args, stdin=None, stdout=subprocess.PIPE, stderr=None)
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
                    simple_progress += newstuff
                    pbar.update(newstuff)
                    if self.vlc_mode and total_duration is not None:
                        print("%d" % int(simple_progress * 100. / total_duration))
                        sys.stdout.flush()
                    if should_print_redirected_stderr:
                        print(pbar_output.read())
                        sys.stdout.flush()
                    in_bytes = np.frombuffer(in_bytes, np.uint8)
                    media_bstring.append(detector(in_bytes))
        self.video_speech_results_ = np.concatenate(media_bstring)
        return self

    def transform(self, *_):
        return self.video_speech_results_


class SubtitleSpeechTransformer(TransformerMixin):
    def __init__(self, sample_rate, start_seconds=0, framerate_ratio=1.):
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
        for sub in subs:
            start = int(round((sub.start.total_seconds() - self.start_seconds) * self.sample_rate))
            duration = sub.end.total_seconds() - sub.start.total_seconds()
            end = start + int(round(duration * self.sample_rate))
            samples[start:end] = min(1. / self.framerate_ratio, 1.)
        self.subtitle_speech_results_ = samples
        return self

    def transform(self, *_):
        return self.subtitle_speech_results_


class DeserializeSpeechTransformer(TransformerMixin):
    def __init__(self):
        self.deserialized_speech_results_ = None

    def fit(self, fname, *_):
        speech = np.load(fname)
        if hasattr(speech, 'files'):
            if 'speech' in speech.files:
                speech = speech['speech']
            else:
                raise ValueError('could not find "speech" array in '
                                 'serialized file; only contains: %s' % speech.files)
        self.deserialized_speech_results_ = speech
        return self

    def transform(self, *_):
        return self.deserialized_speech_results_
