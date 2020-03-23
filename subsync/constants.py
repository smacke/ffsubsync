# -*- coding: utf-8 -*-
SAMPLE_RATE = 100

FRAMERATE_RATIOS = [24./23.976, 25./23.976, 25./24.]

DEFAULT_FRAME_RATE = 48000
DEFAULT_ENCODING = 'infer'
DEFAULT_MAX_SUBTITLE_SECONDS = 10
DEFAULT_START_SECONDS = 0
DEFAULT_SCALE_FACTOR = 1
DEFAULT_VAD = 'subs_then_webrtc'
DEFAULT_MAX_OFFSET_SECONDS = 600

SUBTITLE_EXTENSIONS = ('srt', 'ass', 'ssa')

GITHUB_DEV_USER = 'smacke'
GITHUB_REPO = 'subsync'
DESCRIPTION = 'Synchronize subtitles with video.'
LONG_DESCRIPTION = 'Automatic and language-agnostic synchronization of subtitles with video.'
WEBSITE = 'https://github.com/{}/{}/'.format(GITHUB_DEV_USER, GITHUB_REPO)
DEV_WEBSITE = 'https://smacke.net/'

# No trailing slash important for this one...
API_RELEASE_URL = 'https://api.github.com/repos/{}/{}/releases/latest'.format(GITHUB_DEV_USER, GITHUB_REPO)
RELEASE_URL = 'https://github.com/{}/{}/releases/latest/'.format(GITHUB_DEV_USER, GITHUB_REPO)
