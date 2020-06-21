#!/usr/bin/env python
from ffsubsync.version import __version__


if __name__ == '__main__':
    with open('__version__', 'w') as f:
        f.write(__version__.strip() + '\n')
