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
from auditok import BufferAudioSource, ADSFactory, AudioEnergyValidator, StreamTokenizer
import webrtcvad
from show_progress import show_progress
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

def best_auditok_offset(subtitle_bstring, asegment, sample_rate=100, get_score=False):
    asource = BufferAudioSource(data_buffer=asegment,
                                sampling_rate=48000,
                                sample_width=2,
                                channels=1)
    ads = ADSFactory.ads(audio_source=asource, block_dur=1./sample_rate)
    validator = AudioEnergyValidator(sample_width=asource.get_sample_width(), energy_threshold=50)
    tokenizer = StreamTokenizer(validator=validator, min_length=0.2*sample_rate,
                                max_length=int(5*sample_rate),
                                max_continuous_silence=0.25*sample_rate)
    ads.open()
    tokens = tokenizer.tokenize(ads)
    length = max(token[2] for token in tokens) + 1
    media_bstring = np.zeros(length+1, dtype=int)
    for token in tokens:
        media_bstring[token[1]] += 1
        media_bstring[token[2]+1] -= 1
    media_bstring = (np.cumsum(media_bstring)[:-1] > 0)
    return get_best_offset(subtitle_bstring, media_bstring, get_score=get_score)

def best_webrtcvad_offset(subtitle_bstring, asegment, sample_rate=100, get_score=False):
    vad = webrtcvad.Vad()
    vad.set_mode(3) # set non-speech pruning aggressiveness from 0 to 3
    window_duration = 1./sample_rate # duration in seconds
    samples_per_window = int(window_duration * FRAME_RATE + 0.5)
    bytes_per_sample = 2
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
    media_bstring = np.array(media_bstring)
    return get_best_offset(subtitle_bstring, media_bstring, get_score=get_score)

def get_wav_audio_segment_from_media(fname):
    total_duration = float(ffmpeg.probe(fname)['format']['duration'])
    print('extracting audio...')
    with show_progress(total_duration) as socket_fname:
        out, _ = (
                ffmpeg
                .input(fname)
                .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=FRAME_RATE)
                .global_args('-progress', 'unix://{}'.format(socket_fname))
                .run(capture_stdout=True, capture_stderr=True)
        )
    print('...done')
    return np.frombuffer(out, np.uint8)

def main():
    try:
        reference, subin, subout = [sys.argv[i] for i in range(1,4)]
        subtitle_bstring = binarize_subtitles(subin)
        if reference.endswith('srt'):
            reference_bstring = binarize_subtitles(reference)
            offset_seconds = get_best_offset(subtitle_bstring, reference_bstring) / 100.
        else:
            asegment = get_wav_audio_segment_from_media(reference)
            auditok_out = best_auditok_offset(subtitle_bstring, asegment, get_score=True)
            webrtcvad_out = best_webrtcvad_offset(subtitle_bstring, asegment, get_score=True)
            print('auditok', auditok_out)
            print('webrtcvad', webrtcvad_out)
            offset_seconds = max(auditok_out, webrtcvad_out)[1] / 100.
        print('offset seconds: %.3f' % offset_seconds)
        write_offset_file(subin, subout, offset_seconds)
        return 0
    except Exception as e:
        print(e.stderr, file=sys.stderr)
        return 1

if __name__ == "__main__":
    logging.basicConfig()
    sys.exit(main())
