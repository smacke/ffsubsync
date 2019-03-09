import logging
import sys
from datetime import timedelta

from sklearn.base import TransformerMixin
import srt

from .file_utils import open_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _srt_parse(s, tolerant=True):
    subs = srt.parse(s)
    subs_list = []
    while True:
        try:
            subs_list.append(next(subs))
        except ValueError as e:
            if tolerant:
                logger.warning(e)
                continue
            else:
                raise
        except StopIteration:
            break
    return subs_list


class _SrtMixin(object):
    def __init__(self, subs=None):
        self.subs_ = subs

    def set_encoding(self, encoding):
        self.subs_.set_encoding(encoding)
        return self


class SrtSubtitles(list):
    def __init__(self, *args, **kwargs):
        encoding = kwargs.pop('encoding', None)
        if encoding is None:
            raise ValueError('encoding must be specified')
        super(self.__class__, self).__init__(*args, **kwargs)
        self._encoding = encoding

    def set_encoding(self, encoding):
        if encoding != 'same':
            self._encoding = encoding
        return self

    @property
    def encoding(self):
        return self._encoding

    def write_file(self, fname):
        if sys.version_info[0] > 2:
            with open(fname or sys.stdout.fileno(), 'w', encoding=self.encoding) as f:
                return f.write(srt.compose(self))
        else:
            with (fname and open(fname, 'w')) or sys.stdout as f:
                return f.write(srt.compose(self).encode(self.encoding))


class SrtParser(_SrtMixin, TransformerMixin):
    def __init__(self, encoding='infer'):
        super(self.__class__, self).__init__()
        self.encoding_to_use = encoding

    def fit(self, fname, *_):
        encodings_to_try = (self.encoding_to_use,)
        if self.encoding_to_use == 'infer':
            encodings_to_try = ('utf-8', 'utf-8-sig', 'chinese', 'latin-1')
        with open_file(fname, 'rb') as f:
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
        self.subs_ = SrtSubtitles(offset_subs, encoding=subs.encoding)
        return self

    def transform(self, *_):
        return self.subs_


def read_srt_from_file(fname, encoding='infer'):
    return SrtParser(encoding).fit_transform(fname)


def write_srt_to_file(fname, subs, encoding):
    return SrtSubtitles(subs, encoding=encoding).write_file(fname)


def srt_offset(subs, td_seconds):
    return SrtOffseter(td_seconds).fit_transform(subs)
