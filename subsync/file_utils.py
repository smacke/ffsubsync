# -*- coding: utf-8 -*- 
import six
import sys


class open_file(object):
    """
    Context manager that opens a filename and closes it on exit, but does
    nothing for file-like objects.
    """
    def __init__(self, filename, *args, **kwargs):
        self.closing = kwargs.pop('closing', False)
        if filename is None:
            stream = sys.stdout if 'w' in args else sys.stdin
            if six.PY3:
                self.closeable = open(stream.fileno(), *args, **kwargs)
                self.fh = self.closeable.buffer
            else:
                self.closeable = stream
                self.fh = self.closeable
        elif isinstance(filename, six.string_types):
            self.fh = open(filename, *args, **kwargs)
            self.closeable = self.fh
            self.closing = True
        else:
            self.fh = filename

    def __enter__(self):
        return self.fh

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.closing:
            self.closeable.close()

        return False
