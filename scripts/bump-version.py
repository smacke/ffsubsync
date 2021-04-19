#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import subprocess
import sys

from ffsubsync.version import make_version_tuple


def main(*_):
    components = list(make_version_tuple())
    components[-1] += 1
    version = '.'.join(str(c) for c in components)
    subprocess.check_output(['git', 'tag', version])
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bump version and create git tag.')
    args = parser.parse_args()
    sys.exit(main(args))
