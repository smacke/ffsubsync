# -*- coding: utf-8 -*-
import numpy as np
import pytest
from ffsubsync.aligners import FailedToFindAlignmentException, FFTAligner, MaxScoreAligner


@pytest.mark.parametrize(
    "s1, s2, true_offset",
    [("111001", "11001", -1), ("1001", "1001", 0), ("10010", "01001", 1)],
)
def test_fft_alignment(s1, s2, true_offset):
    assert FFTAligner().fit_transform(s2, s1) == true_offset
    assert MaxScoreAligner(FFTAligner).fit_transform(s2, s1)[0][1] == true_offset
    assert MaxScoreAligner(FFTAligner()).fit_transform(s2, s1)[0][1] == true_offset


@pytest.mark.parametrize(
    "refstring, substring",
    [
        (np.array([]), np.array([1, 0, 1])),  # empty reference
        (np.array([1, 0, 1]), np.array([])),  # empty subtitles
        (np.array([]), np.array([])),  # both empty (would crash math.log)
    ],
)
def test_fft_alignment_rejects_empty_speech(refstring, substring):
    with pytest.raises(FailedToFindAlignmentException, match="empty speech data"):
        FFTAligner().fit(refstring, substring)
