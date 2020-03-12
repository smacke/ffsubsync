# -*- coding: utf-8 -*-
from datetime import timedelta
import logging
import numbers

from sklearn.base import TransformerMixin

from .generic_subtitles import GenericSubtitle, GenericSubtitlesFile, SubsMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubtitleShifter(SubsMixin, TransformerMixin):
    def __init__(self, td_seconds):
        super(SubsMixin, self).__init__()
        if not isinstance(td_seconds, timedelta):
            self.td_seconds = timedelta(seconds=td_seconds)
        else:
            self.td_seconds = td_seconds

    def fit(self, subs, *_):
        self.subs_ = subs.offset(self.td_seconds)
        return self

    def transform(self, *_):
        return self.subs_


class SubtitleScaler(SubsMixin, TransformerMixin):
    def __init__(self, scale_factor):
        super(SubsMixin, self).__init__()
        assert isinstance(scale_factor, numbers.Number)
        self.scale_factor = scale_factor

    def fit(self, subs, *_):
        scaled_subs = []
        for sub in subs:
            scaled_subs.append(
                GenericSubtitle(
                    # py2 doesn't support direct multiplication of timedelta w/ float
                    timedelta(seconds=sub.start.total_seconds() * self.scale_factor),
                    timedelta(seconds=sub.end.total_seconds() * self.scale_factor),
                    sub.inner
                )
            )
        self.subs_ = GenericSubtitlesFile(scaled_subs, format=subs.format, encoding=subs.encoding)
        return self

    def transform(self, *_):
        return self.subs_
