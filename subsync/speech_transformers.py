import logging
import subprocess
import sys
from datetime import timedelta

import ffmpeg
import numpy as np
from sklearn.base import TransformerMixin
import tqdm
import webrtcvad

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    def __init__(self, sample_rate, frame_rate, start_seconds=0, vlc_mode=False):
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.start_seconds = start_seconds
        self.vlc_mode = vlc_mode
        self.video_speech_results_ = None

    def fit(self, fname, *_):
        total_duration = float(ffmpeg.probe(fname)['format']['duration']) - self.start_seconds
        speech_detectors = [_make_webrtcvad_detector(self.sample_rate, self.frame_rate)]
        media_bstrings = [[] for _sd in speech_detectors]
        logger.info('extracting speech segments from video %s...', fname)
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
                if self.vlc_mode:
                    print("%d" % int(simple_progress * 100. / total_duration))
                    sys.stdout.flush()
                in_bytes = np.frombuffer(in_bytes, np.uint8)
                for media_bstring, detector in zip(media_bstrings, speech_detectors):
                    media_bstring.append(detector(in_bytes))
        logger.info('...done.')
        self.video_speech_results_ = [np.concatenate(media_bstring)
                                      for media_bstring in media_bstrings]
        return self

    def transform(self, *_):
        return self.video_speech_results_


class SubtitleSpeechTransformer(TransformerMixin):
    def __init__(self, sample_rate, start_seconds=0):
        self.sample_rate = sample_rate
        self.start_seconds = start_seconds
        self.subtitle_speech_results_ = None
        self.max_time_ = None

    def fit(self, subs, *_):
        logger.info('extracting speech segments from subtitles...')
        max_time = 0
        for sub in subs:
            max_time = max(max_time, sub.end.total_seconds())
        self.max_time_ = max_time - self.start_seconds
        samples = np.zeros(int(max_time * self.sample_rate) + 2, dtype=bool)
        for sub in subs:
            start = int(round((sub.start.total_seconds() - self.start_seconds) * self.sample_rate))
            duration = sub.end.total_seconds() - sub.start.total_seconds()
            end = start + int(round(duration * self.sample_rate))
            samples[start:end] = True
        self.subtitle_speech_results_ = samples
        return self

    def transform(self, *_):
        return self.subtitle_speech_results_
