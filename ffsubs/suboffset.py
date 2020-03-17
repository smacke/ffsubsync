#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import logging
import sys

from sklearn.pipeline import Pipeline

from .subtitle_parser import SrtParser, SrtOffseter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    td = float(sys.argv[3])
    pipe = Pipeline([
        ('parse', SrtParser()),
        ('offset', SrtOffseter(td)),
    ])
    pipe.fit_transform(sys.argv[1])
    pipe.steps[-1][1].write_file(sys.argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
