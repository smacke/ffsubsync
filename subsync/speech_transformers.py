# -*- coding: utf-8 -*- 
import logging
import os
import subprocess
import sys
from datetime import timedelta

import ffmpeg
import numpy as np
from sklearn.base import TransformerMixin
import tqdm
import webrtcvad
try:
    from auditok import \
        BufferAudioSource, ADSFactory, AudioEnergyValidator, StreamTokenizer
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    def __init__(self, vad, sample_rate, frame_rate, start_seconds=0, vlc_mode=False):
        self.vad = vad
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.start_seconds = start_seconds
        self.vlc_mode = vlc_mode
        self.video_speech_results_ = None

    def fit(self, fname, *_):
        try:
            total_duration = float(ffmpeg.probe(fname)['format']['duration']) - self.start_seconds
        except:
            total_duration = None
        if self.vad == 'webrtc':
            detector = _make_webrtcvad_detector(self.sample_rate, self.frame_rate)
        elif self.vad == 'auditok':
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
        with tqdm.tqdm(total=total_duration, disable=self.vlc_mode) as pbar:
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
        self.deserialized_speech_results_ = np.load(fname)
        return self

    def transform(self, *_):
        return self.deserialized_speech_results_
