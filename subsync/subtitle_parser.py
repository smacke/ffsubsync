# -*- coding: utf-8 -*-
from datetime import timedelta
import logging

import cchardet
import pysubs2
from sklearn.base import TransformerMixin
import srt

from .file_utils import open_file
from .generic_subtitles import GenericSubtitle, GenericSubtitlesFile, SubsMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _preprocess_subs(subs, max_subtitle_seconds=None, start_seconds=0, tolerant=True):
    subs_list = []
    start_time = timedelta(seconds=start_seconds)
    max_duration = timedelta(days=1)
    if max_subtitle_seconds is not None:
        max_duration = timedelta(seconds=max_subtitle_seconds)
    subs = iter(subs)
    while True:
        try:
            next_sub = GenericSubtitle.wrap_inner_subtitle(next(subs))
            if next_sub.start < start_time:
                continue
            next_sub.end = min(next_sub.end, next_sub.start + max_duration)
            subs_list.append(next_sub)
        # We don't catch SRTParseError here b/c that is typically raised when we
        # are trying to parse with the wrong encoding, in which case we might
        # be able to try another one on the *entire* set of subtitles elsewhere.
        except ValueError as e:
            if tolerant:
                logger.warning(e)
                continue
            else:
                raise
        except StopIteration:
            break
    return subs_list


class GenericSubtitleParser(SubsMixin, TransformerMixin):
    def __init__(self, fmt='srt', encoding='infer', max_subtitle_seconds=None, start_seconds=0):
        super(self.__class__, self).__init__()
        self.format = fmt
        self.encoding = encoding
        self.detected_encoding_ = None
        self.sub_skippers = []
        self.max_subtitle_seconds = max_subtitle_seconds
        self.start_seconds = start_seconds

    def fit(self, fname, *_):
        encodings_to_try = (self.encoding,)
        with open_file(fname, 'rb') as f:
            subs = f.read()
        if self.encoding == 'infer':
            encodings_to_try = (cchardet.detect(subs)['encoding'],)
        exc = None
        for encoding in encodings_to_try:
            try:
                decoded_subs = subs.decode(encoding, errors='replace').strip()
                if self.format == 'srt':
                    parsed_subs = srt.parse(decoded_subs)
                elif self.format in ('ass', 'ssa'):
                    parsed_subs = pysubs2.SSAFile.from_string(decoded_subs)
                else:
                    raise NotImplementedError('unsupported format: %s' % self.format)
                self.subs_ = GenericSubtitlesFile(
                    _preprocess_subs(parsed_subs,
                                     max_subtitle_seconds=self.max_subtitle_seconds,
                                     start_seconds=self.start_seconds),
                    format=format,
                    encoding=encoding
                )
                self.detected_encoding_ = encoding
                logger.info('Detected encoding: %s' % self.detected_encoding_)
                return self
            except Exception as e:
                exc = e
                continue
        raise exc

    def transform(self, *_):
        return self.subs_
