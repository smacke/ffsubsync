# -*- coding: future_annotations -*-
from typing import List, Tuple


SUBSYNC_RESOURCES_ENV_MAGIC: str = "ffsubsync_resources_xj48gjdkl340"

SAMPLE_RATE: int = 100

FRAMERATE_RATIOS: List[float] = [24./23.976, 25./23.976, 25./24.]

DEFAULT_FRAME_RATE: int = 48000
DEFAULT_NON_SPEECH_LABEL: float = 0.
DEFAULT_ENCODING: str = 'infer'
DEFAULT_MAX_SUBTITLE_SECONDS: int = 10
DEFAULT_START_SECONDS: int = 0
DEFAULT_SCALE_FACTOR: float = 1
DEFAULT_VAD: str = 'subs_then_webrtc'
DEFAULT_MAX_OFFSET_SECONDS: int = 60
DEFAULT_APPLY_OFFSET_SECONDS: int = 0

SUBTITLE_EXTENSIONS: Tuple[str, ...] = ('srt', 'ass', 'ssa', 'sub')

GITHUB_DEV_USER: str = 'smacke'
PROJECT_NAME: str = 'FFsubsync'
PROJECT_LICENSE: str = 'MIT'
COPYRIGHT_YEAR: str = '2019'
GITHUB_REPO: str = 'ffsubsync'
DESCRIPTION: str = 'Synchronize subtitles with video.'
LONG_DESCRIPTION: str = 'Automatic and language-agnostic synchronization of subtitles with video.'
WEBSITE: str = 'https://github.com/{}/{}/'.format(GITHUB_DEV_USER, GITHUB_REPO)
DEV_WEBSITE: str = 'https://smacke.net/'

# No trailing slash important for this one...
API_RELEASE_URL: str = 'https://api.github.com/repos/{}/{}/releases/latest'.format(GITHUB_DEV_USER, GITHUB_REPO)
RELEASE_URL: str = 'https://github.com/{}/{}/releases/latest/'.format(GITHUB_DEV_USER, GITHUB_REPO)
