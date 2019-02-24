#!/usr/bin/env python
import logging
import sys
from datetime import timedelta
import srt


def srt_offset(subs, td_seconds):
    if not isinstance(td_seconds, timedelta):
        td_seconds = timedelta(seconds=td_seconds)
    for sub in subs:
        yield srt.Subtitle(index=sub.index,
                           start=sub.start+td_seconds,
                           end=sub.end+td_seconds,
                           content=sub.content)

def read_srt_from_file(fname):
    with open(fname, 'r') as f:
        return srt.parse(f.read().decode('utf-8'))

def write_srt_to_file(fname, subs):
    with open(fname, 'w') as f:
        f.write(srt.compose(subs).encode('utf-8'))

def main():
    td = float(sys.argv[3])
    subs = read_srt_from_file(sys.argv[1])
    write_srt_to_file(sys.argv[2], srt_offset(subs, timedelta(seconds=td)))
    return 0

if __name__=="__main__":
    logging.basicConfig()
    sys.exit(main())
