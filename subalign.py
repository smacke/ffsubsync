#!/usr/bin/env python
from __future__ import print_function
from builtins import range
import math
import logging
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import ffmpeg
import srt
import tqdm
from auditok import BufferAudioSource, ADSFactory, AudioEnergyValidator, StreamTokenizer
import webrtcvad
from subtimeshift import read_srt_from_file, write_srt_to_file, srt_offset


FRAME_RATE=48000

def get_best_offset(s1, s2, get_score=False):
    a, b = map(lambda s: 2*np.array(s).astype(float) - 1, [s1, s2])
    total_length = int(2**math.ceil(math.log(len(a) + len(b), 2)))
    extra_zeros = total_length - len(a) - len(b)
    convolve = np.fft.ifft(np.fft.fft(np.append(np.zeros(extra_zeros + len(b)), a)) * np.fft.fft(
        np.flip(np.append(b, np.zeros(len(a) + extra_zeros)), 0)))
    convolve = np.real(convolve)
    best_idx = np.argmax(convolve)
    if get_score:
        return convolve[best_idx], len(convolve)-1 - best_idx - len(a)
    else:
        return len(convolve)-1 - best_idx - len(a)

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
        start, end = [int(round(sample_rate * t.total_seconds())) for t in (sub.start, sub.end)]
        samples[start:end+1] = True
    return samples

def make_webrtcvad_detector(sample_rate=100):
    vad = webrtcvad.Vad()
    vad.set_mode(3) # set non-speech pruning aggressiveness from 0 to 3
    window_duration = 1./sample_rate # duration in seconds
    samples_per_window = int(window_duration * FRAME_RATE + 0.5)
    bytes_per_sample = 2
    def _detect(asegment):
        media_bstring = []
        failures = 0
        for start in range(0, len(asegment)//bytes_per_sample, samples_per_window):
            stop = min(start + samples_per_window, len(asegment)//bytes_per_sample)
            try:
                is_speech = vad.is_speech(asegment[start * bytes_per_sample: stop * bytes_per_sample],
                                          sample_rate=FRAME_RATE)
            except:
                is_speech = False
                failures += 1
            media_bstring.append(is_speech)
        return np.array(media_bstring)
    return _detect

def make_auditok_detector(sample_rate=100):
    bytes_per_sample=2
    samples_per_window = FRAME_RATE // sample_rate
    validator = AudioEnergyValidator(sample_width=bytes_per_sample, energy_threshold=50)
    tokenizer = StreamTokenizer(validator=validator, min_length=0.2*sample_rate,
                                max_length=int(5*sample_rate),
                                max_continuous_silence=0.25*sample_rate)
    def _detect(asegment):
        asource = BufferAudioSource(data_buffer=asegment,
                                    sampling_rate=FRAME_RATE,
                                    sample_width=bytes_per_sample,
                                    channels=1)
        ads = ADSFactory.ads(audio_source=asource, block_dur=1./sample_rate)
        ads.open()
        tokens = tokenizer.tokenize(ads)
        #length = max(token[2] for token in tokens) + 1
        length = (len(asegment)//bytes_per_sample + samples_per_window - 1)//samples_per_window
        media_bstring = np.zeros(length+1, dtype=int)
        for token in tokens:
            media_bstring[token[1]] += 1
            media_bstring[token[2]+1] -= 1
        return (np.cumsum(media_bstring)[:-1] > 0)
    return _detect

def get_speech_segments_from_media(fname, *speech_detectors):
    total_duration = float(ffmpeg.probe(fname)['format']['duration'])
    media_bstrings = [[] for _ in speech_detectors]
    print('extracting speech segments...')
    process = (
            ffmpeg
            .input(fname)
            .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=FRAME_RATE)
            .run_async(pipe_stdout=True, quiet=True)
    )
    samples_per_window = 2 * FRAME_RATE // 100
    windows_per_buffer = 10000
    with tqdm.tqdm(total=total_duration) as pbar:
        while True:
            in_bytes = process.stdout.read(samples_per_window * windows_per_buffer)
            if not in_bytes:
                break
            pbar.update(0.5 * len(in_bytes) / FRAME_RATE)
            in_bytes = np.frombuffer(in_bytes, np.uint8)
            for media_bstring, detector in zip(media_bstrings, speech_detectors):
                media_bstring.append(detector(in_bytes))
    print('...done')
    return [np.concatenate(media_bstring) for media_bstring in media_bstrings]

def main():
    try:
        reference, subin, subout = [sys.argv[i] for i in range(1,4)]
        subtitle_bstring = binarize_subtitles(subin)
        if reference.endswith('srt'):
            reference_bstring = binarize_subtitles(reference)
            offset_seconds = get_best_offset(subtitle_bstring, reference_bstring) / 100.
        else:
            auditok_out, webrtcvad_out = get_speech_segments_from_media(
                    reference, make_auditok_detector(), make_webrtcvad_detector())
            auditok_out = get_best_offset(subtitle_bstring, auditok_out, get_score=True)
            webrtcvad_out = get_best_offset(subtitle_bstring, webrtcvad_out, get_score=True)
            print('auditok', auditok_out)
            print('webrtcvad', webrtcvad_out)
            offset_seconds = max(auditok_out, webrtcvad_out)[1] / 100.
        print('offset seconds: %.3f' % offset_seconds)
        write_offset_file(subin, subout, offset_seconds)
        return 0
    except Exception as e:
        print('Exception: %s' % e, file=sys.stderr)
        return 1

if __name__ == "__main__":
    logging.basicConfig()
    sys.exit(main())
