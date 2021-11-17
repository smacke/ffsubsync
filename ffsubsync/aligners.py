# -*- coding: future_annotations -*- 
import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from ffsubsync.golden_section_search import gss
from ffsubsync.sklearn_shim import Pipeline, TransformerMixin

if TYPE_CHECKING:
    from typing import List, Optional, Tuple, Type, Union


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


MIN_FRAMERATE_RATIO = 0.9
MAX_FRAMERATE_RATIO = 1.1


class FailedToFindAlignmentException(Exception):
    pass


class FFTAligner(TransformerMixin):
    def __init__(self, max_offset_samples: Optional[int] = None) -> None:
        self.max_offset_samples: Optional[int] = max_offset_samples
        self.best_offset_: Optional[int] = None
        self.best_score_: Optional[float] = None
        self.get_score_: bool = False

    def _eliminate_extreme_offsets_from_solutions(self, convolve: np.ndarray, substring: np.ndarray) -> np.ndarray:
        convolve = np.copy(convolve)
        if self.max_offset_samples is None:
            return convolve
        offset_to_index = lambda offset: len(convolve) - 1 + offset - len(substring)
        convolve[:offset_to_index(-self.max_offset_samples)] = float('-inf')
        convolve[offset_to_index(self.max_offset_samples):] = float('-inf')
        return convolve

    def _compute_argmax(self, convolve: np.ndarray, substring: np.ndarray) -> None:
        best_idx = np.argmax(convolve)
        self.best_offset_ = len(convolve) - 1 - best_idx - len(substring)
        self.best_score_ = convolve[best_idx]

    def fit(self, refstring, substring, get_score: bool = False) -> FFTAligner:
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
        self._compute_argmax(self._eliminate_extreme_offsets_from_solutions(convolve, substring), substring)
        self.get_score_ = get_score
        return self

    def transform(self, *_) -> Union[int, Tuple[float, int]]:
        if self.get_score_:
            return self.best_score_, self.best_offset_
        else:
            return self.best_offset_


class MaxScoreAligner(TransformerMixin):
    def __init__(
        self,
        base_aligner: Union[FFTAligner, Type[FFTAligner]],
        srtin: Optional[str] = None,
        sample_rate=None,
        max_offset_seconds=None
    ) -> None:
        self.srtin: Optional[str] = srtin
        if sample_rate is None or max_offset_seconds is None:
            self.max_offset_samples: Optional[int] = None
        else:
            self.max_offset_samples = abs(int(max_offset_seconds * sample_rate))
        if isinstance(base_aligner, type):
            self.base_aligner: FFTAligner = base_aligner(max_offset_samples=self.max_offset_samples)
        else:
            self.base_aligner = base_aligner
        self.max_offset_seconds: Optional[int] = max_offset_seconds
        self._scores: List[Tuple[Tuple[float, int], Pipeline]] = []

    def fit_gss(self, refstring, subpipe_maker):
        def opt_func(framerate_ratio, is_last_iter):
            subpipe = subpipe_maker(framerate_ratio)
            substring = subpipe.fit_transform(self.srtin)
            score = self.base_aligner.fit_transform(refstring, substring, get_score=True)
            logger.info('got score %.0f (offset %d) for ratio %.3f', score[0], score[1], framerate_ratio)
            if is_last_iter:
                self._scores.append((score, subpipe))
            return -score[0]
        gss(opt_func, MIN_FRAMERATE_RATIO, MAX_FRAMERATE_RATIO)
        return self

    def fit(self, refstring, subpipes: Union[Pipeline, List[Pipeline]]) -> MaxScoreAligner:
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

    def transform(self, *_) -> Tuple[Tuple[float, float], Pipeline]:
        scores = self._scores
        if self.max_offset_samples is not None:
            scores = list(filter(lambda s: abs(s[0][1]) <= self.max_offset_samples, scores))
        if len(scores) == 0:
            raise FailedToFindAlignmentException('Synchronization failed; consider passing '
                                                 '--max-offset-seconds with a number larger than '
                                                 '{}'.format(self.max_offset_seconds))
        (score, offset), subpipe = max(scores, key=lambda x: x[0][0])
        return (score, offset), subpipe
