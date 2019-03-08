import logging
import sys
from datetime import timedelta

from sklearn.base import TransformerMixin
import srt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _srt_parse(s, tolerant=True):
    subs = srt.parse(s)
    subs_list = []
    while True:
        try:
            subs_list.append(next(subs))
        except ValueError:
            if tolerant:
                continue
            else:
                raise
        except StopIteration:
            break
    return subs_list


class _SrtMixin(object):
    def __init__(self, subs=None):
        self.subs_ = subs


class SrtSubtitles(list):
    def __init__(self, *args, encoding=None, **kwargs):
        if encoding is None:
            raise ValueError('encoding must be specified')
        super(self.__class__, self).__init__(*args, **kwargs)
        self._encoding = encoding

    def write_file(self, fname):
        if sys.version_info[0] > 2:
            with open(fname or sys.stdout.fileno(), 'w', encoding=self._encoding) as f:
                return f.write(srt.compose(self))
        else:
            with (fname and open(fname, 'w')) or sys.stdout as f:
                return f.write(srt.compose(self).encode(self._encoding))


class SrtParser(_SrtMixin, TransformerMixin):
    def __init__(self, encoding='infer'):
        super(self.__class__, self).__init__()
        self.encoding_to_use = encoding

    def fit(self, fname, *_):
        encodings_to_try = (self.encoding_to_use,)
        if self.encoding_to_use == 'infer':
            encodings_to_try = ('utf-8', 'utf-8-sig', 'chinese', 'latin-1')
        if sys.version_info[0] > 2:
            with open(fname or sys.stdin.fileno(), 'r') as f:
                subs = f.buffer.read()
        else:
            with (fname and open(fname, 'r')) or sys.stdin as f:
                subs = f.read()
        exc = None
        for encoding in encodings_to_try:
            try:
                self.subs_ = SrtSubtitles(
                    _srt_parse(subs.decode(encoding).strip()),
                    encoding=encoding
                )
                return self
            except Exception as e:
                exc = e
                continue
        raise exc

    def transform(self, *_):
        return self.subs_


class SrtOffseter(_SrtMixin, TransformerMixin):
    def __init__(self, td_seconds):
        super(_SrtMixin, self).__init__()
        if not isinstance(td_seconds, timedelta):
            self.td_seconds = timedelta(seconds=td_seconds)
        else:
            self.td_seconds = td_seconds

    def fit(self, subs, *_):
        offset_subs = []
        for sub in subs:
            offset_subs.append(srt.Subtitle(index=sub.index,
                                            start=sub.start + self.td_seconds,
                                            end=sub.end + self.td_seconds,
                                            content=sub.content))
        self.subs_ = SrtSubtitles(offset_subs, encoding=subs._encoding)
        return self

    def transform(self, *_):
        return self.subs_


def read_srt_from_file(fname, encoding='infer'):
    return SrtParser(encoding).fit_transform(fname)


def write_srt_to_file(fname, subs, encoding):
    return SrtSubtitles(subs, encoding=encoding).write_file(fname)


def srt_offset(subs, td_seconds):
    return SrtOffseter(td_seconds).fit_transform(subs)
