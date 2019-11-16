# -*- coding: utf-8 -*- 
import logging
import math

import numpy as np
from sklearn.base import TransformerMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FFTAligner(TransformerMixin):
    def __init__(self):
        self.best_offset_ = None
        self.best_score_ = None
        self.get_score_ = False

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
        best_idx = np.argmax(convolve)
        self.best_offset_ = len(convolve) - 1 - best_idx - len(substring)
        self.best_score_ = convolve[best_idx]
        self.get_score_ = get_score
        return self

    def transform(self, *_):
        if self.get_score_:
            return self.best_score_, self.best_offset_
        else:
            return self.best_offset_


class MaxScoreAligner(TransformerMixin):
    def __init__(self, base_aligner):
        if isinstance(base_aligner, type):
            self.base_aligner = base_aligner()
        else:
            self.base_aligner = base_aligner
        self._scores = []

    def fit(self, refstring, subpipes):
        logger.info('computing alignments...')
        if not isinstance(subpipes, list):
            subpipes = [subpipes]
        for subpipe in subpipes:
            if hasattr(subpipe, 'transform'):
                substring = subpipe.transform()
            else:
                substring = subpipe
            self._scores.append((
                self.base_aligner.fit_transform(
                    refstring, substring, get_score=True
                ),
                subpipe
            ))
        logger.info('...done')
        return self

    def transform(self, *_):
        (score, offset), subpipe = max(self._scores)
        return offset, subpipe
