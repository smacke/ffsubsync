# -*- coding: utf-8 -*-
import re
from typing import List, Optional, Tuple


SUBSYNC_RESOURCES_ENV_MAGIC: str = "ffsubsync_resources_xj48gjdkl340"

SAMPLE_RATE: int = 100

FRAMERATE_RATIOS: List[float] = [24.0 / 23.976, 25.0 / 23.976, 25.0 / 24.0]

DEFAULT_FRAME_RATE: int = 48000
DEFAULT_NON_SPEECH_LABEL: float = 0.0
DEFAULT_ENCODING: str = "infer"
DEFAULT_MAX_SUBTITLE_SECONDS: int = 10
DEFAULT_START_SECONDS: int = 0
DEFAULT_SCALE_FACTOR: float = 1
DEFAULT_VAD: str = "subs_then_webrtc"
DEFAULT_MAX_OFFSET_SECONDS: int = 60
DEFAULT_APPLY_OFFSET_SECONDS: int = 0

# Penalty applied by the alass-style piecewise aligner (--split-penalty) when the
# flag is given without an explicit value. Expressed in seconds-of-overlap: a split
# is only introduced if it increases total speech overlap by more than this many
# seconds, which suppresses spurious per-cue splits while still admitting genuine
# structural breaks (ad breaks / scene cuts) that misalign everything after them.
DEFAULT_SPLIT_PENALTY: float = 5.0

# Weight of the piecewise aligner's edge/length term ("standard scoring"): each cue
# is additionally charged this fraction of the reference speech in a guard band just
# outside its edges, so a cue prefers a same-sized speech block over sitting anywhere
# inside a longer one. 0 disables it (pure overlap scoring).
DEFAULT_SPLIT_LENGTH_PENALTY: float = 0.25

# Sub-sample resolution of the piecewise aligner's offset search: the offset grid is
# refined to 1/this of a sample (1 => whole samples, i.e. 1000/SAMPLE_RATE ms steps).
# The reference is a step function, so finer offsets are computed by exact linear
# interpolation of its prefix sum. 1 is plenty for perceptual sync; raise it only when
# sub-(1000/SAMPLE_RATE)ms precision actually matters.
DEFAULT_SPLIT_SUBSAMPLE: int = 1

# The named voice activity detectors accepted by --vad. Kept here (rather than
# inline in the argparse `choices=`) so the CLI help text and the manual
# validation in validate_args share a single source of truth -- necessary
# because --vad no longer uses argparse `choices` (in whisper-transcription mode
# it may instead carry a path to a VAD model; see WhisperSpeechTransformer).
VAD_CHOICES: Tuple[str, ...] = (
    "subs_then_webrtc",
    "webrtc",
    "subs_then_auditok",
    "auditok",
    "subs_then_silero",
    "silero",
    "fused",
    "fused:weighted",
    "fused:intersection",
    "fused:union",
)

# The auditok detector additionally accepts a parameter after a colon (also
# with a "subs_then_" prefix): a number is a fixed energy threshold in dB
# ("auditok:50" reproduces the historical behavior), "auto"/"otsu" estimate
# the threshold from the audio itself (bare "auditok" is an alias for
# "auditok:auto"), "pXX" places it at the XXth percentile of window energies,
# and "webrtc[:N]" runs auditok's event segmentation over WebRTC VAD frame
# decisions (N = aggressiveness, 0-3). These parameterized forms cannot be
# enumerated in VAD_CHOICES, hence the regex + helper used by validate_args.
_AUDITOK_SPEC_RE = re.compile(
    r"^(auto|otsu|percentile|p[1-9][0-9]?|webrtc(:[0-3])?|[0-9]+(\.[0-9]+)?)$"
)


def is_valid_vad(vad: str) -> bool:
    """Return True if *vad* is a valid --vad value, including the
    parameterized ``auditok:<spec>`` forms."""
    if vad in VAD_CHOICES:
        return True
    base = vad[len("subs_then_") :] if vad.startswith("subs_then_") else vad
    if base.startswith("auditok:"):
        return _AUDITOK_SPEC_RE.match(base[len("auditok:") :]) is not None
    return False

# ffmpeg's whisper audio filter (a reference source alternative to audio VAD;
# see WhisperSpeechTransformer). "auto" lets whisper detect the language; the
# queue default is bumped above ffmpeg's own 3s since larger windows give more
# accurate cue timings (what we actually align against) at the cost of CPU.
DEFAULT_WHISPER_LANGUAGE: str = "auto"
DEFAULT_WHISPER_QUEUE: int = 8
# internal vad-mode token used when transcription is selected via --whisper-weights
WHISPER_VAD: str = "whisper"

# Quality gating (--skip-sync-on-low-quality). The score's sign is meaningful even
# though its magnitude is not, so 0.0 rejects only anti-correlated alignments. The
# framerate-deviation default clears every real correction (discrete ratios reach
# ~0.0427 and a typical --gss result stays well within +/-0.1), so it does not
# reject a legitimate correction -- it is a knob to tighten when you know the
# framerate should not change.
DEFAULT_MIN_SCORE: float = 0.0
DEFAULT_QUALITY_MAX_OFFSET_SECONDS: float = 30.0
DEFAULT_MAX_FRAMERATE_DEVIATION: float = 0.1

SUBTITLE_EXTENSIONS: Tuple[str, ...] = ("srt", "ass", "ssa", "sub")

# Remote URL protocols that ffmpeg can read directly as an input (`-i <url>`),
# so a reference of this form needs no local download or read-permission check.
REMOTE_URL_PROTOCOLS: Tuple[str, ...] = (
    "http://",
    "https://",
    "rtmp://",
    "rtsp://",
    "ftp://",
)


def is_remote_url(path: Optional[str]) -> bool:
    """Return True if *path* is a remote URL ffmpeg can stream directly."""
    if path is None:
        return False
    return path.startswith(REMOTE_URL_PROTOCOLS)


GITHUB_DEV_USER: str = "smacke"
PROJECT_NAME: str = "FFsubsync"
PROJECT_LICENSE: str = "MIT"
COPYRIGHT_YEAR: str = "2019"
GITHUB_REPO: str = "ffsubsync"
DESCRIPTION: str = "Synchronize subtitles with video."
LONG_DESCRIPTION: str = (
    "Automatic and language-agnostic synchronization of subtitles with video."
)
WEBSITE: str = "https://github.com/{}/{}/".format(GITHUB_DEV_USER, GITHUB_REPO)
DEV_WEBSITE: str = "https://smacke.net/"

# No trailing slash important for this one...
API_RELEASE_URL: str = "https://api.github.com/repos/{}/{}/releases/latest".format(
    GITHUB_DEV_USER, GITHUB_REPO
)
RELEASE_URL: str = "https://github.com/{}/{}/releases/latest/".format(
    GITHUB_DEV_USER, GITHUB_REPO
)
