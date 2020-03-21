#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys

from gooey import Gooey, GooeyParser

from .subsync import run, add_cli_only_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@Gooey(tabbed_groups=True)
def make_parser():
    parser = GooeyParser(description='Synchronize subtitles with video.')
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
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
