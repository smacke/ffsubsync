# -*- coding: utf-8 -*-
from datetime import timedelta
import logging
from typing import Any, Optional

try:
    import cchardet as chardet
except ImportError:
    import chardet  # type: ignore
import pysubs2
from ffsubsync.sklearn_shim import TransformerMixin
import srt

from ffsubsync.constants import *
from ffsubsync.file_utils import open_file
from ffsubsync.generic_subtitles import GenericSubtitle, GenericSubtitlesFile, SubsMixin

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def _preprocess_subs(
    subs,
    max_subtitle_seconds: Optional[int] = None,
    start_seconds: int = 0,
    tolerant: bool = True,
) -> List[GenericSubtitle]:
    subs_list = []
    start_time = timedelta(seconds=start_seconds)
    max_duration = timedelta(days=1)
    if max_subtitle_seconds is not None:
        max_duration = timedelta(seconds=max_subtitle_seconds)
    subs = iter(subs)
    while True:
        try:
            next_sub = GenericSubtitle.wrap_inner_subtitle(next(subs))
            if next_sub.start < start_time:
                continue
            next_sub.end = min(next_sub.end, next_sub.start + max_duration)
            subs_list.append(next_sub)
        # We don't catch SRTParseError here b/c that is typically raised when we
        # are trying to parse with the wrong encoding, in which case we might
        # be able to try another one on the *entire* set of subtitles elsewhere.
        except ValueError as e:
            if tolerant:
                logger.warning(e)
                continue
            else:
                raise
        except StopIteration:
            break
    return subs_list


class GenericSubtitleParser(SubsMixin, TransformerMixin):
    def __init__(
        self,
        fmt: str = "srt",
        encoding: str = "infer",
        caching: bool = False,
        max_subtitle_seconds: Optional[int] = None,
        start_seconds: int = 0,
        skip_ssa_info: bool = False,
    ) -> None:
        super(self.__class__, self).__init__()
        self.sub_format: str = fmt
        self.encoding: str = encoding
        self.caching: bool = caching
        self.fit_fname: Optional[str] = None
        self.detected_encoding_: Optional[str] = None
        self.max_subtitle_seconds: Optional[int] = max_subtitle_seconds
        self.start_seconds: int = start_seconds
        # FIXME: hack to get tests to pass; remove
        self._skip_ssa_info: bool = skip_ssa_info

    def fit(self, fname: str, *_) -> "GenericSubtitleParser":
        if self.caching and self.fit_fname == ("<stdin>" if fname is None else fname):
            return self
        encodings_to_try = (self.encoding,)
        with open_file(fname, "rb") as f:
            subs = f.read()
        if self.encoding == "infer":
            encodings_to_try = (chardet.detect(subs)["encoding"],)
            self.detected_encoding_ = encodings_to_try[0]
            logger.info("detected encoding: %s" % self.detected_encoding_)
        exc = None
        for encoding in encodings_to_try:
            try:
                decoded_subs = subs.decode(encoding, errors="replace").strip()
                if self.sub_format == "srt":
                    parsed_subs = srt.parse(decoded_subs, ignore_errors=True)
                elif self.sub_format in ("ass", "ssa", "sub"):
                    parsed_subs = pysubs2.SSAFile.from_string(decoded_subs)
                else:
                    raise NotImplementedError(
                        "unsupported format: %s" % self.sub_format
                    )
                extra_generic_subtitle_file_kwargs = {}
                if isinstance(parsed_subs, pysubs2.SSAFile):
                    extra_generic_subtitle_file_kwargs.update(
                        dict(
                            styles=parsed_subs.styles,
                            # pysubs2 on Python >= 3.6 doesn't support this
                            fonts_opaque=getattr(parsed_subs, "fonts_opaque", None),
                            info=parsed_subs.info if not self._skip_ssa_info else None,
                        )
                    )
                self.subs_ = GenericSubtitlesFile(
                    _preprocess_subs(
                        parsed_subs,
                        max_subtitle_seconds=self.max_subtitle_seconds,
                        start_seconds=self.start_seconds,
                    ),
                    sub_format=self.sub_format,
                    encoding=encoding,
                    **extra_generic_subtitle_file_kwargs,
                )
                self.fit_fname = "<stdin>" if fname is None else fname
                if len(encodings_to_try) > 1:
                    self.detected_encoding_ = encoding
                    logger.info("detected encoding: %s" % self.detected_encoding_)
                return self
            except Exception as e:
                exc = e
                continue
        raise exc

    def transform(self, *_) -> GenericSubtitlesFile:
        return self.subs_


def make_subtitle_parser(
    fmt: str,
    encoding: str = DEFAULT_ENCODING,
    caching: bool = False,
    max_subtitle_seconds: int = DEFAULT_MAX_SUBTITLE_SECONDS,
    start_seconds: int = DEFAULT_START_SECONDS,
    **kwargs: Any,
) -> GenericSubtitleParser:
    return GenericSubtitleParser(
        fmt=fmt,
        encoding=encoding,
        caching=caching,
        max_subtitle_seconds=max_subtitle_seconds,
        start_seconds=start_seconds,
        skip_ssa_info=kwargs.get("skip_ssa_info", False),
    )
