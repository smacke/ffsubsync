#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys

from gooey import Gooey, GooeyParser
import requests
from requests.exceptions import Timeout

from .constants import API_RELEASE_URL, RELEASE_URL, WEBSITE, DEV_WEBSITE, DESCRIPTION, LONG_DESCRIPTION
from .subsync import run, add_cli_only_args
from .version import __version__

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _make_version_tuple(vstr):
    if vstr[0] == 'v':
        vstr = vstr[1:]
    return tuple(map(int, vstr.split('.')))


def _update_available():
    try:
        resp = requests.get(API_RELEASE_URL, timeout=1)
        latest_vstr = resp.json()['tag_name']
    except Timeout:
        return False
    except KeyError:
        return False
    if not resp.ok:
        return False
    return _make_version_tuple(__version__) < _make_version_tuple(latest_vstr)


_menu = [
    {
        'name': 'File',
        'items': [
            {
                'type': 'AboutDialog',
                'menuTitle': 'About',
                'name': 'Subsync',
                'description': LONG_DESCRIPTION,
                'version': __version__,
                'copyright': '2019',
                'website': WEBSITE,
                'developer': DEV_WEBSITE,
                'license': 'MIT'
            },
            {
                'type': 'Link',
                'menuTitle': 'Check for updates',
                'url': RELEASE_URL,
            }
        ]
    }
]


@Gooey(
    menu=_menu,
    tabbed_groups=True,
    progress_regex=r"(\d+)%",
    hide_progress_msg=True
)
def make_parser():
    description = DESCRIPTION
    if _update_available():
        description += '\nUpdate available! Please go to "File" -> "Check for updates" to update Subsync.'
    parser = GooeyParser(description=description)
    main_group = parser.add_argument_group('Required Options')
    main_group.add_argument(
        'reference',
        help='Reference (video or subtitles file) to which to synchronize input subtitles.',
        widget='FileChooser'
    )
    main_group.add_argument('srtin', help='Input subtitles file', widget='FileChooser')
    main_group.add_argument('-o', '--srtout', help='Output subtitles file (default=${srtin}.synced.srt).')
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument('--merge-with-reference', '--merge', action='store_true',
                                help='Merge reference subtitles with synced output subtitles.')
    advanced_group.add_argument('--make-test-case', '--create-test-case', action='store_true',
                                help='If specified, create a test archive a few KiB in size '
                                     'to send to the developer as a debugging aid.')
    return parser


def main():
    parser = make_parser()
    _ = parser.parse_args()  # Fool Gooey into presenting the simpler menu
    add_cli_only_args(parser)
    args = parser.parse_args()
    args.gui_mode = True
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
