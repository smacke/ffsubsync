import logging
import sys
import threading

import ffmpeg
import numpy as np
from sklearn.base import TransformerMixin
import tqdm
import webrtcvad

from .utils import read_srt_from_file

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
    def __init__(self, sample_rate, frame_rate, progress_only=False):
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.progress_only = progress_only

    def transform(self, fname, *_):
        total_duration = float(ffmpeg.probe(fname)['format']['duration'])
        speech_detectors = [_make_webrtcvad_detector(self.sample_rate, self.frame_rate)]
        media_bstrings = [[] for _ in speech_detectors]
        logger.info('extracting speech segments from video %s...', fname)
        process = (
            ffmpeg.input(fname)
            .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=self.frame_rate)
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        threading.Thread(target=lambda: process.stderr.read()).start()
        bytes_per_frame = 2
        frames_per_window = bytes_per_frame * self.frame_rate // self.sample_rate
        windows_per_buffer = 10000
        simple_progress = 0.
        with tqdm.tqdm(total=total_duration, disable=self.progress_only) as pbar:
            while True:
                in_bytes = process.stdout.read(frames_per_window * windows_per_buffer)
                if not in_bytes:
                    break
                newstuff = len(in_bytes) / float(bytes_per_frame) / self.frame_rate
                simple_progress += newstuff
                pbar.update(newstuff)
                if self.progress_only:
                    print("%d" % int(simple_progress * 100. / total_duration))
                    sys.stdout.flush()
                in_bytes = np.frombuffer(in_bytes, np.uint8)
                for media_bstring, detector in zip(media_bstrings, speech_detectors):
                    media_bstring.append(detector(in_bytes))
        logger.info('...done.')
        return [np.concatenate(media_bstring) for media_bstring in media_bstrings]

    def fit(self, *_):
        return self


class SubtitleSpeechTransformer(TransformerMixin):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def transform(self, fname, *_):
        logger.info('extracting speech segments from subtitles %s...', fname)
        max_time = 0
        for sub in read_srt_from_file(fname):
            max_time = max(max_time, sub.end.total_seconds())
        samples = np.zeros(int(max_time * self.sample_rate) + 2, dtype=bool)
        for sub in read_srt_from_file(fname):
            start, end = [int(round(self.sample_rate * t.total_seconds()))
                          for t in (sub.start, sub.end)]
            samples[start:end + 1] = True
        return samples

    def fit(self, *_):
        return self
