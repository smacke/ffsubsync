#!/usr/bin/env python
from __future__ import print_function
from builtins import range
import argparse
import math
import logging
import sys
import threading
import numpy as np
import ffmpeg
import tqdm
import webrtcvad
from .utils import read_srt_from_file, write_srt_to_file, srt_offset
from .version import __version__

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRAME_RATE = 48000


def get_best_offset(substring, vidstring, get_score=False):
    substring, vidstring = [
        list(map(int, s))
        if isinstance(s, str) else s
        for s in [substring, vidstring]
    ]
    substring, vidstring = map(
        lambda s: 2 * np.array(s).astype(float) - 1, [substring, vidstring])
    total_bits = math.log(len(substring) + len(vidstring), 2)
    total_length = int(2 ** math.ceil(total_bits))
    extra_zeros = total_length - len(substring) - len(vidstring)
    subft = np.fft.fft(np.append(np.zeros(extra_zeros + len(vidstring)), substring))
    vidft = np.fft.fft(np.flip(np.append(vidstring, np.zeros(len(substring) + extra_zeros)), 0))
    convolve = np.real(np.fft.ifft(subft * vidft))
    best_idx = np.argmax(convolve)
    if get_score:
        return convolve[best_idx], len(convolve) - 1 - best_idx - len(substring)
    else:
        return len(convolve) - 1 - best_idx - len(substring)


def write_offset_file(fread, fwrite, nseconds):
    subs = read_srt_from_file(fread)
    subs = srt_offset(subs, nseconds)
    write_srt_to_file(fwrite, subs)


def binarize_subtitles(fname, sample_rate=100):
    max_time = 0
    for sub in read_srt_from_file(fname):
        max_time = max(max_time, sub.end.total_seconds())
    samples = np.zeros(int(max_time * sample_rate) + 2, dtype=bool)
    for sub in read_srt_from_file(fname):
        start, end = [int(round(sample_rate * t.total_seconds()))
                      for t in (sub.start, sub.end)]
        samples[start:end + 1] = True
    return samples


def make_webrtcvad_detector(sample_rate=100):
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # set non-speech pruning aggressiveness from 0 to 3
    window_duration = 1. / sample_rate  # duration in seconds
    frames_per_window = int(window_duration * FRAME_RATE + 0.5)
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
                    sample_rate=FRAME_RATE)
            except:
                is_speech = False
                failures += 1
            # webrtcvad has low recall on this setting, so treat non-speech as "not sure"
            media_bstring.append(1. if is_speech else 0.5)
        return np.array(media_bstring)

    return _detect


def get_speech_segments_from_media(fname, progress_only, *speech_detectors):
    total_duration = float(ffmpeg.probe(fname)['format']['duration'])
    media_bstrings = [[] for _ in speech_detectors]
    logger.info('extracting speech segments...')
    process = (
        ffmpeg.input(fname)
        .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=FRAME_RATE)
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    threading.Thread(target=lambda: process.stderr.read()).start()
    bytes_per_frame = 2
    sample_rate = 100
    frames_per_window = bytes_per_frame * FRAME_RATE // sample_rate
    windows_per_buffer = 10000
    simple_progress = 0.
    with tqdm.tqdm(total=total_duration, disable=progress_only) as pbar:
        while True:
            in_bytes = process.stdout.read(
                frames_per_window * windows_per_buffer)
            if not in_bytes:
                break
            newstuff = len(in_bytes) / float(bytes_per_frame) / FRAME_RATE
            simple_progress += newstuff
            pbar.update(newstuff)
            if progress_only:
                print("%d" % int(simple_progress * 100. / total_duration))
                sys.stdout.flush()
            in_bytes = np.frombuffer(in_bytes, np.uint8)
            for media_bstring, detector in zip(
                    media_bstrings, speech_detectors):
                media_bstring.append(detector(in_bytes))
    logger.info('...done.')
    return [np.concatenate(media_bstring) for media_bstring in media_bstrings]


def main():
    parser = argparse.ArgumentParser(
        description='Synchronize subtitles with video.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))
    parser.add_argument('reference')
    parser.add_argument('-i', '--srtin', required=True)  # TODO: allow read from stdin
    parser.add_argument('-o', '--srtout', default=None)
    parser.add_argument('--progress-only', action='store_true')
    args = parser.parse_args()
    if args.progress_only:
        logger.setLevel(logging.CRITICAL)
    subtitle_bstring = binarize_subtitles(args.srtin)
    if args.reference.endswith('srt'):
        reference_bstring = binarize_subtitles(args.reference)
        offset_seconds = get_best_offset(
            subtitle_bstring, reference_bstring) / 100.
    else:
        (webrtcvad_out,) = get_speech_segments_from_media(
            args.reference,
            args.progress_only,
            make_webrtcvad_detector()
        )
        logger.info('computing alignments...')
        webrtcvad_out = get_best_offset(
            subtitle_bstring, webrtcvad_out, get_score=True)
        logger.info('...done.')
        logger.info('webrtcvad: %s', webrtcvad_out)
        offset_seconds = webrtcvad_out[1] / 100.
    logger.info('offset seconds: %.3f', offset_seconds)
    if not (args.progress_only and args.srtout is None):
        write_offset_file(args.srtin, args.srtout, offset_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
