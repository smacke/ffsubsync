# -*- coding: utf-8 -*-
import logging
import sys

try:
    from rich.console import Console
    from rich.logging import RichHandler

    # configure logging here because some other later imported library does it first otherwise
    # TODO: use a fileconfig
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=Console(file=sys.stderr))],
    )
except:  # noqa: E722
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

from .version import __version__  # noqa
from .ffsubsync import main  # noqa
