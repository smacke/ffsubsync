#!/usr/bin/env python
import math
import logging
import os
import sys
from datetime import datetime, timedelta
import tempfile
import numpy as np
from scipy.io import wavfile
import sh
import srt
from auditok import BufferAudioSource, ADSFactory, AudioEnergyValidator, StreamTokenizer
import webrtcvad
from subtimeshift import read_srt_from_file, write_srt_to_file, srt_offset


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
    samples = []
    prev_end = timedelta(seconds=0)
    total_time = 0
    for sub in read_srt_from_file(fname):
        if prev_end >= sub.start:
            continue
        total_time += (sub.end - sub.start).total_seconds()
        samples.extend([0]*int(round(sample_rate * (sub.start - prev_end).total_seconds())))
        samples.extend([1]*int(round(sample_rate * (sub.end - sub.start).total_seconds())))
        prev_end = sub.end
    return np.array(samples).astype(bool)

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
    frame_rate = 48000
    vad = webrtcvad.Vad()
    vad.set_mode(3) # set non-speech pruning aggressiveness from 0 to 3
    window_duration = 1./sample_rate # duration in seconds
    print 'num bytes', len(asegment)
    samples_per_window = int(window_duration * frame_rate + 0.5)
    print 'samples per window', samples_per_window
    bytes_per_sample = 2
    media_bstring = []
    failures = 0
    for start in xrange(0, len(asegment)/bytes_per_sample, samples_per_window):
        stop = min(start + samples_per_window, len(asegment)/bytes_per_sample)
        try:
            is_speech = vad.is_speech(asegment[start * bytes_per_sample: stop * bytes_per_sample],
                                      sample_rate=frame_rate)
        except:
            is_speech = False
            failures += 1
        #media_bstring.extend([is_speech]*(stop-start))
        media_bstring.append(is_speech)
    print 'failures', failures
    media_bstring = np.array(media_bstring)
    return get_best_offset(subtitle_bstring, media_bstring, get_score=get_score)

def get_wav_audio_segment_from_media(fname):
    with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
        wavname = tmp.name
    try:
        sh.ffmpeg('-i', fname, '-vn', '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '1', wavname)
        frame_rate, samples = wavfile.read(wavname)
        if frame_rate != 48000:
            raise Exception('Unexpected frame rate: %d' % frame_rate)
        #return struct.pack('%dh' % len(samples), *samples)
        return samples.data
    finally:
        os.unlink(wavname)

def main():
    reference, subin, subout = [sys.argv[i] for i in xrange(1,4)]
    subtitle_bstring = binarize_subtitles(subin)
    if reference.endswith('srt'):
        reference_bstring = binarize_subtitles(reference)
        offset_seconds = get_best_offset(subtitle_bstring, reference_bstring) / 100.
    else:
        asegment = get_wav_audio_segment_from_media(reference)
        auditok_out = best_auditok_offset(subtitle_bstring, asegment, get_score=True)
        webrtcvad_out = best_webrtcvad_offset(subtitle_bstring, asegment, get_score=True)
        print 'auditok', auditok_out
        print 'webrtcvad', webrtcvad_out
        offset_seconds = max(auditok_out, webrtcvad_out)[1] / 100.
    print 'offset seconds: %.3f' % offset_seconds
    write_offset_file(subin, subout, offset_seconds)
    return 0

if __name__ == "__main__":
    logging.basicConfig()
    sys.exit(main())
