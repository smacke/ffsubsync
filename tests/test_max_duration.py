# -*- coding: utf-8 -*-
from datetime import timedelta

from ffsubsync.constants import DEFAULT_FRAME_RATE, DEFAULT_VAD, SAMPLE_RATE
from ffsubsync.ffsubsync import make_parser, make_reference_pipe
from ffsubsync.speech_transformers import VideoSpeechTransformer


def _transformer(**overrides):
    kwargs = dict(
        vad="webrtc",
        sample_rate=SAMPLE_RATE,
        frame_rate=DEFAULT_FRAME_RATE,
        non_speech_label=0.0,
    )
    kwargs.update(overrides)
    return VideoSpeechTransformer(**kwargs)


def test_no_max_duration_omits_t_flag():
    args = _transformer()._build_ffmpeg_args("movie.mkv")
    assert "-t" not in args


def test_max_duration_adds_input_side_t_flag():
    args = _transformer(max_duration_seconds=600)._build_ffmpeg_args("movie.mkv")
    # -t must precede -i so ffmpeg limits how much input it reads/downloads
    assert "-t" in args
    t_idx = args.index("-t")
    i_idx = args.index("-i")
    assert t_idx < i_idx
    assert args[t_idx + 1] == str(timedelta(seconds=600))


def test_max_duration_t_value_is_formatted_like_start_seconds():
    # both -ss and -t use timedelta formatting, so fractional seconds survive
    args = _transformer(
        start_seconds=5, max_duration_seconds=12.5
    )._build_ffmpeg_args("movie.mkv")
    assert args[args.index("-t") + 1] == str(timedelta(seconds=12.5))
    assert args[args.index("-ss") + 1] == str(timedelta(seconds=5))


def test_cli_parses_max_duration_seconds():
    args = make_parser().parse_args(
        ["movie.mkv", "--max-duration-seconds", "600"]
    )
    assert args.max_duration_seconds == 600.0
    # default is None (process the whole reference)
    assert make_parser().parse_args(["movie.mkv"]).max_duration_seconds is None


def test_make_reference_pipe_threads_max_duration_into_transformer():
    args = make_parser().parse_args(
        ["movie.mkv", "--vad", DEFAULT_VAD, "--max-duration-seconds", "120"]
    )
    pipe = make_reference_pipe(args)
    transformer = pipe.named_steps["speech_extract"]
    assert isinstance(transformer, VideoSpeechTransformer)
    assert transformer.max_duration_seconds == 120.0
