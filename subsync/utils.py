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

def srt_parse(s, tolerant=True):
    subs = srt.parse(s)
    subs_list = []
    while True:
        try:
            subs_list.append(next(subs))
        except ValueError:
            if tolerant:
                continue
            else:
                raise
        except StopIteration:
            break
    return subs_list

def read_srt_from_file(fname, encoding='infer'):
    encodings_to_try = (encoding,)
    if encoding == 'infer':
        encodings_to_try = ('utf-8', 'utf-8-sig', 'latin-1')
    subs = None
    if fname is None:
        if sys.version_info[0] > 2:
            subs = sys.stdin.buffer.read()
        else:
            subs = sys.stdin.read()
    exc = None
    for encoding in encodings_to_try:
        try:
            if subs is not None:
                return srt_parse(subs.decode(encoding))
            else:
                if sys.version_info[0] > 2:
                    with open(fname, 'r', encoding=encoding) as f:
                        return srt_parse(f.read())
                else:
                    with open(fname, 'r') as f:
                        return srt_parse(f.read().decode(encoding))
        except Exception as e:
            exc = e
            continue
    raise exc

def write_srt_to_file(fname, subs):
    if fname is None:
        return sys.stdout.write(srt.compose(subs))
    if sys.version_info[0] > 2:
        with open(fname, 'w', encoding='utf-8') as f:
            return f.write(srt.compose(subs))
    else:
        with open(fname, 'w') as f:
            return f.write(srt.compose(subs).encode('utf-8'))

def main():
    td = float(sys.argv[3])
    subs = read_srt_from_file(sys.argv[1])
    write_srt_to_file(sys.argv[2], srt_offset(subs, timedelta(seconds=td)))
    return 0

if __name__=="__main__":
    logging.basicConfig()
    sys.exit(main())
