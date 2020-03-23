#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys

from gooey import Gooey, GooeyParser

from .constants import (
    RELEASE_URL,
    WEBSITE,
    DEV_WEBSITE,
    DESCRIPTION,
    LONG_DESCRIPTION,
    PROJECT_NAME,
    PROJECT_LICENSE,
    COPYRIGHT_YEAR,
)
from .subsync import run, add_cli_only_args
from .version import __version__, update_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_menu = [
    {
        'name': 'File',
        'items': [
            {
                'type': 'AboutDialog',
                'menuTitle': 'About',
                'name': PROJECT_NAME,
                'description': LONG_DESCRIPTION,
                'version': __version__,
                'copyright': COPYRIGHT_YEAR,
                'website': WEBSITE,
                'developer': DEV_WEBSITE,
                'license': PROJECT_LICENSE,
            },
            {
                'type': 'Link',
                'menuTitle': 'Download latest release',
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
    if update_available():
        description += '\nUpdate available! Please go to "File" -> "Download latest release" to update Subsync.'
    parser = GooeyParser(description=description)
    main_group = parser.add_argument_group('Required Options')
    main_group.add_argument(
        'reference',
        help='Reference (video or subtitles file) to which to synchronize input subtitles.',
        widget='FileChooser'
    )
    main_group.add_argument('srtin', help='Input subtitles file', widget='FileChooser')
    main_group.add_argument('-o', '--srtout',
                            help='Output subtitles file (default=${srtin}.synced.srt).',
                            widget='FileSaver')
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
