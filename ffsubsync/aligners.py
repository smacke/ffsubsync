# -*- coding: utf-8 -*- 
import logging
import math

import numpy as np

from .constants import FRAMERATE_RATIOS
from .golden_section_search import gss
from .sklearn_shim import TransformerMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailedToFindAlignmentException(Exception):
    pass


class FFTAligner(TransformerMixin):
    def __init__(self, max_offset_samples=None):
        self.max_offset_samples = max_offset_samples
        self.best_offset_ = None
        self.best_score_ = None
        self.get_score_ = False

    def _zero_out_extreme_offsets(self, convolve, substring):
        convolve = np.copy(convolve)
        if self.max_offset_samples is None:
            return convolve
        offset_to_index = lambda offset: len(convolve) - 1 + offset - len(substring)
        convolve[:offset_to_index(-self.max_offset_samples)] = convolve[offset_to_index(self.max_offset_samples):] = 0
        return convolve

    def _compute_argmax(self, convolve, substring):
        best_idx = np.argmax(convolve)
        self.best_offset_ = len(convolve) - 1 - best_idx - len(substring)
        self.best_score_ = convolve[best_idx]

    def fit(self, refstring, substring, get_score=False):
        refstring, substring = [
            list(map(int, s))
            if isinstance(s, str) else s
            for s in [refstring, substring]
        ]
        refstring, substring = map(
            lambda s: 2 * np.array(s).astype(float) - 1, [refstring, substring])
        total_bits = math.log(len(substring) + len(refstring), 2)
        total_length = int(2 ** math.ceil(total_bits))
        extra_zeros = total_length - len(substring) - len(refstring)
        subft = np.fft.fft(np.append(np.zeros(extra_zeros + len(refstring)), substring))
        refft = np.fft.fft(np.flip(np.append(refstring, np.zeros(len(substring) + extra_zeros)), 0))
        convolve = np.real(np.fft.ifft(subft * refft))
        self._compute_argmax(self._zero_out_extreme_offsets(convolve, substring), substring)
        if self.best_score_ == 0.:
            self._compute_argmax(convolve, substring)
        self.get_score_ = get_score
        return self

    def transform(self, *_):
        if self.get_score_:
            return self.best_score_, self.best_offset_
        else:
            return self.best_offset_


class MaxScoreAligner(TransformerMixin):
    def __init__(self, base_aligner, srtin=None, sample_rate=None, max_offset_seconds=None):
        self.srtin = srtin
        if sample_rate is None or max_offset_seconds is None:
            self.max_offset_samples = None
        else:
            self.max_offset_samples = abs(int(max_offset_seconds * sample_rate))
        if isinstance(base_aligner, type):
            self.base_aligner = base_aligner(max_offset_samples=self.max_offset_samples)
        else:
            self.base_aligner = base_aligner
        self.max_offset_seconds = max_offset_seconds
        self._scores = []

    def fit_gss(self, refstring, subpipe_maker):
        def opt_func(framerate_ratio, is_last_iter):
            subpipe = subpipe_maker(framerate_ratio)
            substring = subpipe.fit_transform(self.srtin)
            score = self.base_aligner.fit_transform(refstring, substring, get_score=True)
            logger.info('got score %.0f (offset %d) for ratio %.3f', score[0], score[1], framerate_ratio)
            if is_last_iter:
                self._scores.append((score, subpipe))
            return -score[0]
        gss(opt_func, 0.9, 1.1)
        return self

    def fit(self, refstring, subpipes):
        if not isinstance(subpipes, list):
            subpipes = [subpipes]
        for subpipe in subpipes:
            if callable(subpipe):
                self.fit_gss(refstring, subpipe)
                continue
            elif hasattr(subpipe, 'transform'):
                substring = subpipe.transform(self.srtin)
            else:
                substring = subpipe
            self._scores.append((
                self.base_aligner.fit_transform(
                    refstring, substring, get_score=True
                ),
                subpipe
            ))
        return self

    def transform(self, *_):
        scores = self._scores
        if self.max_offset_samples is not None:
            scores = list(filter(lambda s: abs(s[0][1]) <= self.max_offset_samples, scores))
        if len(scores) == 0:
            raise FailedToFindAlignmentException('Synchronization failed; consider passing '
                                                 '--max-offset-seconds with a number larger than '
                                                 '{}'.format(self.max_offset_seconds))
        (score, offset), subpipe = max(scores, key=lambda x: x[0][0])
        return (score, offset), subpipe
