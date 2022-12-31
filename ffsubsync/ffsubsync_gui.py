#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import sys

from gooey import Gooey, GooeyParser

from ffsubsync.constants import (
    RELEASE_URL,
    WEBSITE,
    DEV_WEBSITE,
    DESCRIPTION,
    LONG_DESCRIPTION,
    PROJECT_NAME,
    PROJECT_LICENSE,
    COPYRIGHT_YEAR,
    SUBSYNC_RESOURCES_ENV_MAGIC,
)

# set the env magic so that we look for resources in the right place
if SUBSYNC_RESOURCES_ENV_MAGIC not in os.environ:
    os.environ[SUBSYNC_RESOURCES_ENV_MAGIC] = getattr(sys, "_MEIPASS", "")
from ffsubsync.ffsubsync import run, add_cli_only_args
from ffsubsync.version import get_version, update_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_menu = [
    {
        "name": "File",
        "items": [
            {
                "type": "AboutDialog",
                "menuTitle": "About",
                "name": PROJECT_NAME,
                "description": LONG_DESCRIPTION,
                "version": get_version(),
                "copyright": COPYRIGHT_YEAR,
                "website": WEBSITE,
                "developer": DEV_WEBSITE,
                "license": PROJECT_LICENSE,
            },
            {
                "type": "Link",
                "menuTitle": "Download latest release",
                "url": RELEASE_URL,
            },
        ],
    }
]


@Gooey(
    program_name=PROJECT_NAME,
    image_dir=os.path.join(os.environ[SUBSYNC_RESOURCES_ENV_MAGIC], "img"),
    menu=_menu,
    tabbed_groups=True,
    progress_regex=r"(\d+)%",
    hide_progress_msg=True,
)
def make_parser():
    description = DESCRIPTION
    if update_available():
        description += (
            "\nUpdate available! Please go to "
            '"File" -> "Download latest release"'
            " to update FFsubsync."
        )
    parser = GooeyParser(description=description)
    main_group = parser.add_argument_group("Basic")
    main_group.add_argument(
        "reference",
        help="Reference (video or subtitles file) to which to synchronize input subtitles.",
        widget="FileChooser",
    )
    main_group.add_argument("srtin", help="Input subtitles file", widget="FileChooser")
    main_group.add_argument(
        "-o",
        "--srtout",
        help="Output subtitles file (default=${srtin}.synced.srt).",
        widget="FileSaver",
    )
    advanced_group = parser.add_argument_group("Advanced")

    # TODO: these are shared between gui and cli; don't duplicate this code
    advanced_group.add_argument(
        "--merge-with-reference",
        "--merge",
        action="store_true",
        help="Merge reference subtitles with synced output subtitles.",
    )
    advanced_group.add_argument(
        "--make-test-case",
        "--create-test-case",
        action="store_true",
        help="If specified, create a test archive a few KiB in size "
        "to send to the developer as a debugging aid.",
    )
    advanced_group.add_argument(
        "--reference-stream",
        "--refstream",
        "--reference-track",
        "--reftrack",
        default=None,
        help="Which stream/track in the video file to use as reference, "
        "formatted according to ffmpeg conventions. For example, s:0 "
        "uses the first subtitle track; a:3 would use the fourth audio track.",
    )
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
