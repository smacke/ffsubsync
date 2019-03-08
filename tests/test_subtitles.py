import sys
from io import BytesIO

import pytest
from sklearn.pipeline import make_pipeline

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


@pytest.mark.parametrize('encoding', [
    'utf-8',
    'ascii',
    'latin-1',
])
def test_same_encoding(encoding):
    parser = SrtParser(encoding=encoding)
    offseter = SrtOffseter(1)
    pipe = make_pipeline(parser, offseter)
    pipe.fit(BytesIO(fake_srt))
    assert parser.subs_.encoding == encoding
    assert offseter.subs_.encoding == parser.subs_.encoding
