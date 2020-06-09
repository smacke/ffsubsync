# -*- coding: utf-8 -*-
from datetime import timedelta
import logging

import cchardet
import pysubs2
from .sklearn_shim import TransformerMixin
import srt

from .constants import *
from .file_utils import open_file
from .generic_subtitles import GenericSubtitle, GenericSubtitlesFile, SubsMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_subtitle_parser(
        fmt,
        encoding=DEFAULT_ENCODING,
        caching=False,
        max_subtitle_seconds=DEFAULT_MAX_SUBTITLE_SECONDS,
        start_seconds=DEFAULT_START_SECONDS,
        **kwargs
):
    return GenericSubtitleParser(
        fmt=fmt,
        encoding=encoding,
        caching=caching,
        max_subtitle_seconds=max_subtitle_seconds,
        start_seconds=start_seconds
    )


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
    def __init__(self, fmt='srt', encoding='infer', caching=False, max_subtitle_seconds=None, start_seconds=0):
        super(self.__class__, self).__init__()
        self.sub_format = fmt
        self.encoding = encoding
        self.caching = caching
        self.fit_fname = None
        self.detected_encoding_ = None
        self.sub_skippers = []
        self.max_subtitle_seconds = max_subtitle_seconds
        self.start_seconds = start_seconds

    def fit(self, fname, *_):
        if self.caching and self.fit_fname == fname:
            return self
        encodings_to_try = (self.encoding,)
        with open_file(fname, 'rb') as f:
            subs = f.read()
        if self.encoding == 'infer':
            encodings_to_try = (cchardet.detect(subs)['encoding'],)
            self.detected_encoding_ = encodings_to_try[0]
            logger.info('detected encoding: %s' % self.detected_encoding_)
        exc = None
        for encoding in encodings_to_try:
            try:
                decoded_subs = subs.decode(encoding, errors='replace').strip()
                if self.sub_format == 'srt':
                    parsed_subs = srt.parse(decoded_subs)
                elif self.sub_format in ('ass', 'ssa', 'sub'):
                    parsed_subs = pysubs2.SSAFile.from_string(decoded_subs)
                else:
                    raise NotImplementedError('unsupported format: %s' % self.sub_format)
                self.subs_ = GenericSubtitlesFile(
                    _preprocess_subs(parsed_subs,
                                     max_subtitle_seconds=self.max_subtitle_seconds,
                                     start_seconds=self.start_seconds),
                    sub_format=self.sub_format,
                    encoding=encoding
                )
                self.fit_fname = fname
                if len(encodings_to_try) > 1:
                    self.detected_encoding_ = encoding
                    logger.info('detected encoding: %s' % self.detected_encoding_)
                return self
            except Exception as e:
                exc = e
                continue
        raise exc

    def transform(self, *_):
        return self.subs_
