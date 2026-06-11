# -*- coding: utf-8 -*-
from datetime import timedelta

import numpy as np
import pytest
import srt

from ffsubsync.constants import SAMPLE_RATE
from ffsubsync.generic_subtitles import GenericSubtitle
from ffsubsync.speech_transformers import SubtitleSpeechTransformer, _is_metadata


@pytest.mark.parametrize(
    "content",
    [
        "[music]",
        "(applause)",
        "{laughter}",
        "（音乐）",  # full-width brackets
        "【掌声】",
        "「効果音」",
        "♪",
        "♪♪",
        "♪ ♫ ♬",
        "<i>[music]</i>",  # cue wrapped in markup
        "<font color='#fff'>(applause)</font>",
        "<i></i>",  # markup with no text
        "   ",  # whitespace only
        "",
    ],
)
def test_is_metadata_detects_non_dialogue(content):
    assert _is_metadata(content, False) is True


@pytest.mark.parametrize(
    "content",
    [
        "John: hello there",
        "<i>Hello?</i>",  # markup-wrapped dialogue must NOT be metadata
        "♪ We are the champions ♪",  # lyrics with words are kept
        "[door creaks] and he walks in",  # bracketed aside + real dialogue
        "(5 > 3) is true",
    ],
)
def test_is_metadata_keeps_dialogue(content):
    assert _is_metadata(content, False) is False


@pytest.mark.parametrize(
    "content",
    ["this is in english", "Bob - Alice"],
)
def test_is_metadata_boundary_only_heuristics(content):
    # "english" / " - " are only treated as metadata at the very start or end
    assert _is_metadata(content, True) is True
    assert _is_metadata(content, False) is False


def _generic_sub(start, end, content):
    inner = srt.Subtitle(
        index=1,
        start=timedelta(seconds=start),
        end=timedelta(seconds=end),
        content=content,
    )
    return GenericSubtitle(timedelta(seconds=start), timedelta(seconds=end), inner)


def test_non_dialogue_line_contributes_no_speech_but_is_not_dropped():
    subs = [
        _generic_sub(0, 1, "♪ ♫"),  # cue: should be zeroed in the signal
        _generic_sub(1, 2, "Hello there"),  # dialogue: should be speech
    ]
    transformer = SubtitleSpeechTransformer(sample_rate=SAMPLE_RATE).fit(subs)
    speech = transformer.subtitle_speech_results_
    # the cue's frames are non-speech; the dialogue's frames are speech
    assert not np.any(speech[: SAMPLE_RATE])
    assert np.all(speech[SAMPLE_RATE : 2 * SAMPLE_RATE] > 0)
    # the transformer never drops subtitles; it only affects the speech signal
    assert len(subs) == 2
