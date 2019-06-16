import itertools
from io import BytesIO
from datetime import timedelta

import pytest
import numpy as np
from sklearn.pipeline import make_pipeline

from subsync.speech_transformers import SubtitleSpeechTransformer
from subsync.subtitle_parsers import SrtParser, SrtOffseter

fake_srt = b"""1
00:00:00,178 --> 00:00:02,416
<i>Previously on "Your favorite TV show..."</i>

2
00:00:02,828 --> 00:00:04,549
Oh hi, Mark.

3
00:00:04,653 --> 00:00:06,062
You are tearing me apart, Lisa!
"""


@pytest.mark.parametrize('start_seconds', [0, 2, 4, 6])
def test_start_seconds(start_seconds):
    parser_zero = SrtParser(start_seconds=0)
    parser_zero.fit(BytesIO(fake_srt))
    parser = SrtParser(start_seconds=start_seconds)
    parser.fit(BytesIO(fake_srt))
    expected = [sub for sub in parser_zero.subs_ if sub.start >= timedelta(seconds=start_seconds)]
    assert expected == parser.subs_


@pytest.mark.parametrize('max_seconds', [1, 1.5, 2.0, 2.5])
def test_max_seconds(max_seconds):
    parser = SrtParser(max_subtitle_seconds=max_seconds)
    parser.fit(BytesIO(fake_srt))
    assert max(sub.end - sub.start for sub in parser.subs_) <= timedelta(seconds=max_seconds)


@pytest.mark.parametrize('encoding', ['utf-8', 'ascii', 'latin-1'])
def test_same_encoding(encoding):
    parser = SrtParser(encoding=encoding)
    offseter = SrtOffseter(1)
    pipe = make_pipeline(parser, offseter)
    pipe.fit(BytesIO(fake_srt))
    assert parser.subs_.encoding == encoding
    assert offseter.subs_.encoding == parser.subs_.encoding
    assert offseter.subs_.set_encoding('same').encoding == encoding
    assert offseter.subs_.set_encoding('utf-8').encoding == 'utf-8'


@pytest.mark.parametrize('offset', [1, 1.5, -2.3])
def test_offset(offset):
    parser = SrtParser()
    offseter = SrtOffseter(offset)
    pipe = make_pipeline(parser, offseter)
    pipe.fit(BytesIO(fake_srt))
    for sub_orig, sub_offset in zip(parser.subs_, offseter.subs_):
        assert abs(sub_offset.start.total_seconds() -
                   sub_orig.start.total_seconds() - offset) < 1e-6
        assert abs(sub_offset.end.total_seconds() -
                   sub_orig.end.total_seconds() - offset) < 1e-6


@pytest.mark.parametrize('sample_rate,start_seconds',
                         itertools.product([10, 20, 100, 300], [0, 2, 4, 6]))
def test_speech_extraction(sample_rate, start_seconds):
    parser = SrtParser(start_seconds=start_seconds)
    extractor = SubtitleSpeechTransformer(sample_rate=sample_rate, start_seconds=start_seconds)
    pipe = make_pipeline(parser, extractor)
    bitstring = pipe.fit_transform(BytesIO(fake_srt))
    bitstring_shifted_left = np.append(bitstring[1:], [False])
    bitstring_shifted_right = np.append([False], bitstring[:-1])
    bitstring_cumsum = np.cumsum(bitstring)
    consec_ones_end_pos = np.nonzero(bitstring_cumsum *
                                     (bitstring ^ bitstring_shifted_left) *
                                     (bitstring_cumsum != np.cumsum(bitstring_shifted_right)))[0]
    prev = 0
    for pos, sub in zip(consec_ones_end_pos, parser.subs_):
        start = int(round(sub.start.total_seconds() * sample_rate))
        duration = sub.end.total_seconds() - sub.start.total_seconds()
        stop = start + int(round(duration * sample_rate))
        assert bitstring_cumsum[pos] - prev == stop - start
        prev = bitstring_cumsum[pos]


def test_max_time_found():
    parser = SrtParser()
    extractor = SubtitleSpeechTransformer(sample_rate=100)
    pipe = make_pipeline(parser, extractor)
    pipe.fit(BytesIO(fake_srt))
    assert extractor.max_time_ == 6.062
