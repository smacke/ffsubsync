#!/usr/bin/env python
from __future__ import print_function
from builtins import range

import argparse
import math
import logging
import sys

import numpy as np

from .speech_transformers import SubtitleSpeechTransformer, VideoSpeechTransformer
from .utils import read_srt_from_file, write_srt_to_file, srt_offset
from .version import __version__

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRAME_RATE = 48000
SAMPLE_RATE = 100


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
    subtitle_bstring = SubtitleSpeechTransformer(
        sample_rate=SAMPLE_RATE).fit_transform(args.srtin)
    if args.reference.endswith('srt'):
        reference_bstring = SubtitleSpeechTransformer(
            sample_rate=SAMPLE_RATE).fit_transform(args.reference)
    else:
        (reference_bstring,) = VideoSpeechTransformer(
            sample_rate=SAMPLE_RATE,
            frame_rate=FRAME_RATE,
            progress_only=args.progress_only).fit_transform(args.reference)
    logger.info('computing alignments...')
    offset_seconds = get_best_offset(subtitle_bstring, reference_bstring) / 100.
    logger.info('offset seconds: %.3f', offset_seconds)
    write_offset_file(args.srtin, args.srtout, offset_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
