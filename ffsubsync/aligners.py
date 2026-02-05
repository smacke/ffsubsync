# -*- coding: utf-8 -*-
import logging
import math
from typing import List, Optional, Tuple, Type, Union, Dict

import numpy as np

from ffsubsync.golden_section_search import gss
from ffsubsync.sklearn_shim import Pipeline, TransformerMixin


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


MIN_FRAMERATE_RATIO = 0.9
MAX_FRAMERATE_RATIO = 1.1


class FailedToFindAlignmentException(Exception):
    pass


def compute_weighted_median_offset(
    segment_results: List[Dict],
    min_score: float = 0.3,
    max_offset_variance: float = 5.0
) -> Tuple[float, float, List[Dict]]:
    """Compute weighted median offset from multiple segment alignment results.
    
    Args:
        segment_results: List of dicts with 'offset_seconds', 'score', 'segment_start' keys.
        min_score: Minimum score threshold to include a segment.
        max_offset_variance: Maximum allowed variance in offsets (seconds) before warning.
        
    Returns:
        Tuple of (median_offset_seconds, combined_score, filtered_results).
        
    Raises:
        FailedToFindAlignmentException: If no valid segments found.
    """
    # Filter segments by score
    valid_segments = [s for s in segment_results if s.get('score', 0) >= min_score]
    
    if len(valid_segments) == 0:
        # Try with lower threshold
        valid_segments = [s for s in segment_results if s.get('score', 0) > 0]
        if len(valid_segments) == 0:
            raise FailedToFindAlignmentException(
                "Multi-segment sync failed: no segments with positive score"
            )
        logger.warning(
            "No segments with score >= %.2f, using %d segments with lower scores",
            min_score, len(valid_segments)
        )
    
    # Extract offsets and scores
    offsets = np.array([s['offset_seconds'] for s in valid_segments])
    scores = np.array([s['score'] for s in valid_segments])
    
    # Check variance
    offset_variance = np.var(offsets)
    if offset_variance > max_offset_variance:
        logger.warning(
            "High variance in segment offsets (%.2f): segments may have different time shifts. "
            "Consider using single-segment sync or checking video source.",
            offset_variance
        )
    
    # Compute weighted median
    # Sort by offset
    sorted_indices = np.argsort(offsets)
    sorted_offsets = offsets[sorted_indices]
    sorted_scores = scores[sorted_indices]
    
    # Weighted median: find the offset where cumulative weight >= 50%
    total_weight = np.sum(sorted_scores)
    cumulative_weight = np.cumsum(sorted_scores)
    median_idx = np.searchsorted(cumulative_weight, total_weight / 2)
    median_idx = min(median_idx, len(sorted_offsets) - 1)
    
    median_offset = sorted_offsets[median_idx]
    combined_score = np.mean(scores)  # Average score
    
    logger.info(
        "Multi-segment sync: %d/%d valid segments, median_offset=%.3fs, avg_score=%.3f, variance=%.3f",
        len(valid_segments), len(segment_results), median_offset, combined_score, offset_variance
    )
    
    # Add detailed log
    for s in valid_segments:
        logger.debug(
            "  Segment @%ds: offset=%.3fs, score=%.3f",
            s.get('segment_start', 0), s['offset_seconds'], s['score']
        )
    
    return median_offset, combined_score, valid_segments


class FFTAligner(TransformerMixin):
    def __init__(self, max_offset_samples: Optional[int] = None) -> None:
        self.max_offset_samples: Optional[int] = max_offset_samples
        self.best_offset_: Optional[int] = None
        self.best_score_: Optional[float] = None
        self.get_score_: bool = False

    def _eliminate_extreme_offsets_from_solutions(
        self, convolve: np.ndarray, substring: np.ndarray
    ) -> np.ndarray:
        convolve = np.copy(convolve)
        if self.max_offset_samples is None:
            return convolve

        def _offset_to_index(offset):
            return len(convolve) - 1 + offset - len(substring)

        convolve[: _offset_to_index(-self.max_offset_samples)] = float("-inf")
        convolve[_offset_to_index(self.max_offset_samples) :] = float("-inf")
        return convolve

    def _compute_argmax(self, convolve: np.ndarray, substring: np.ndarray) -> None:
        best_idx = int(np.argmax(convolve))
        self.best_offset_ = len(convolve) - 1 - best_idx - len(substring)
        self.best_score_ = convolve[best_idx]

    def fit(self, refstring, substring, get_score: bool = False) -> "FFTAligner":
        refstring, substring = [
            list(map(int, s)) if isinstance(s, str) else s
            for s in [refstring, substring]
        ]
        # Check for empty arrays before FFT
        if len(refstring) == 0 or len(substring) == 0:
            raise FailedToFindAlignmentException(
                "Cannot align empty speech data (refstring=%d, substring=%d)" 
                % (len(refstring), len(substring))
            )
        refstring, substring = map(
            lambda s: 2 * np.array(s).astype(float) - 1, [refstring, substring]
        )
        total_bits = math.log(len(substring) + len(refstring), 2)
        total_length = int(2 ** math.ceil(total_bits))
        extra_zeros = total_length - len(substring) - len(refstring)
        subft = np.fft.fft(np.append(np.zeros(extra_zeros + len(refstring)), substring))
        refft = np.fft.fft(
            np.flip(np.append(refstring, np.zeros(len(substring) + extra_zeros)), 0)
        )
        convolve = np.real(np.fft.ifft(subft * refft))
        self._compute_argmax(
            self._eliminate_extreme_offsets_from_solutions(convolve, substring),
            substring,
        )
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
        max_offset_seconds=None,
    ) -> None:
        self.srtin: Optional[str] = srtin
        if sample_rate is None or max_offset_seconds is None:
            self.max_offset_samples: Optional[int] = None
        else:
            self.max_offset_samples = abs(int(max_offset_seconds * sample_rate))
        if isinstance(base_aligner, type):
            self.base_aligner: FFTAligner = base_aligner(
                max_offset_samples=self.max_offset_samples
            )
        else:
            self.base_aligner = base_aligner
        self.max_offset_seconds: Optional[int] = max_offset_seconds
        self._scores: List[Tuple[Tuple[float, int], Pipeline]] = []

    def fit_gss(self, refstring, subpipe_maker):
        def opt_func(framerate_ratio, is_last_iter):
            subpipe = subpipe_maker(framerate_ratio)
            substring = subpipe.fit_transform(self.srtin)
            score = self.base_aligner.fit_transform(
                refstring, substring, get_score=True
            )
            logger.info(
                "got score %.0f (offset %d) for ratio %.3f",
                score[0],
                score[1],
                framerate_ratio,
            )
            if is_last_iter:
                self._scores.append((score, subpipe))
            return -score[0]

        gss(opt_func, MIN_FRAMERATE_RATIO, MAX_FRAMERATE_RATIO)
        return self

    def fit(
        self, refstring, subpipes: Union[Pipeline, List[Pipeline]]
    ) -> "MaxScoreAligner":
        if not isinstance(subpipes, list):
            subpipes = [subpipes]
        for subpipe in subpipes:
            if callable(subpipe):
                self.fit_gss(refstring, subpipe)
                continue
            elif hasattr(subpipe, "transform"):
                substring = subpipe.transform(self.srtin)
            else:
                substring = subpipe
            self._scores.append(
                (
                    self.base_aligner.fit_transform(
                        refstring, substring, get_score=True
                    ),
                    subpipe,
                )
            )
        return self

    def transform(self, *_) -> Tuple[Tuple[float, float], Pipeline]:
        scores = self._scores
        if self.max_offset_samples is not None:
            scores = list(
                filter(lambda s: abs(s[0][1]) <= self.max_offset_samples, scores)
            )
        if len(scores) == 0:
            raise FailedToFindAlignmentException(
                "Synchronization failed; consider passing "
                "--max-offset-seconds with a number larger than "
                "{}".format(self.max_offset_seconds)
            )
        (score, offset), subpipe = max(scores, key=lambda x: x[0][0])
        return (score, offset), subpipe


class TwoPassAligner(TransformerMixin):
    """Two-pass alignment for improved accuracy.
    
    First pass: Coarse alignment with large search range
    Second pass: Fine alignment with small search range around coarse result
    """
    
    def __init__(
        self,
        sample_rate: int,
        coarse_max_offset_seconds: float = 60.0,
        fine_max_offset_seconds: float = 5.0
    ) -> None:
        self.sample_rate = sample_rate
        self.coarse_max_offset_seconds = coarse_max_offset_seconds
        self.fine_max_offset_seconds = fine_max_offset_seconds
        self.coarse_offset_: Optional[int] = None
        self.coarse_score_: Optional[float] = None
        self.fine_offset_: Optional[int] = None
        self.fine_score_: Optional[float] = None
        self.total_offset_: Optional[int] = None
        self.final_score_: Optional[float] = None
    
    def fit(self, refstring: np.ndarray, substring: np.ndarray, get_score: bool = False) -> "TwoPassAligner":
        """Perform two-pass alignment.
        
        Args:
            refstring: Reference speech signal
            substring: Subtitle speech signal
            get_score: Whether to return score (always True for this aligner)
        """
        # First pass: Coarse alignment
        coarse_max_samples = int(self.coarse_max_offset_seconds * self.sample_rate)
        coarse_aligner = FFTAligner(max_offset_samples=coarse_max_samples)
        coarse_aligner.fit(refstring, substring, get_score=True)
        self.coarse_score_, self.coarse_offset_ = coarse_aligner.transform()
        
        logger.info(
            "Two-pass alignment - coarse: score=%.1f, offset=%d samples (%.2fs)",
            self.coarse_score_, self.coarse_offset_, self.coarse_offset_ / self.sample_rate
        )
        
        # Apply coarse offset to substring
        if self.coarse_offset_ >= 0:
            shifted_substring = np.concatenate([
                np.zeros(self.coarse_offset_),
                substring[:len(substring) - self.coarse_offset_] if self.coarse_offset_ < len(substring) else []
            ])
        else:
            abs_offset = abs(self.coarse_offset_)
            shifted_substring = np.concatenate([
                substring[abs_offset:] if abs_offset < len(substring) else [],
                np.zeros(abs_offset)
            ])
        
        # Ensure shifted_substring has same length as original
        if len(shifted_substring) < len(substring):
            shifted_substring = np.concatenate([shifted_substring, np.zeros(len(substring) - len(shifted_substring))])
        elif len(shifted_substring) > len(substring):
            shifted_substring = shifted_substring[:len(substring)]
        
        # Second pass: Fine alignment with smaller range
        fine_max_samples = int(self.fine_max_offset_seconds * self.sample_rate)
        fine_aligner = FFTAligner(max_offset_samples=fine_max_samples)
        fine_aligner.fit(refstring, shifted_substring, get_score=True)
        self.fine_score_, self.fine_offset_ = fine_aligner.transform()
        
        logger.info(
            "Two-pass alignment - fine: score=%.1f, offset=%d samples (%.2fs)",
            self.fine_score_, self.fine_offset_, self.fine_offset_ / self.sample_rate
        )
        
        # Combine offsets
        self.total_offset_ = self.coarse_offset_ + self.fine_offset_
        self.final_score_ = self.fine_score_
        
        logger.info(
            "Two-pass alignment - total: score=%.1f, offset=%d samples (%.2fs)",
            self.final_score_, self.total_offset_, self.total_offset_ / self.sample_rate
        )
        
        return self
    
    def transform(self, *_) -> Tuple[float, int]:
        return self.final_score_, self.total_offset_
