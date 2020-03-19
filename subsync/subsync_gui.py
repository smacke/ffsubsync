#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys

from gooey import Gooey, GooeyParser

from .subsync import run, add_gui_and_cli_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@Gooey
def make_parser():
    parser = GooeyParser(description='Synchronize subtitles with video.')
    add_gui_and_cli_args(parser, mode='gui')
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    return run(args, mode='gui')


if __name__ == "__main__":
    sys.exit(main())
