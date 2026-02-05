FFsubsync
=======

[![CI Status](https://github.com/smacke/ffsubsync/workflows/ffsubsync/badge.svg)](https://github.com/smacke/ffsubsync/actions)
[![Support Ukraine](https://badgen.net/badge/support/UKRAINE/?color=0057B8&labelColor=FFD700)](https://github.com/vshymanskyy/StandWithUkraine/blob/main/docs/README.md)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-maroon.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/ffsubsync.svg)](https://pypi.org/project/ffsubsync)
[![Documentation Status](https://readthedocs.org/projects/ffsubsync/badge/?version=latest)](https://ffsubsync.readthedocs.io/en/latest/?badge=latest)
[![PyPI Version](https://img.shields.io/pypi/v/ffsubsync.svg)](https://pypi.org/project/ffsubsync)


Language-agnostic automatic synchronization of subtitles with video, so that
subtitles are aligned to the correct starting point within the video.

Turn this:                       |  Into this:
:-------------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/smacke/ffsubsync/master/resources/img/tearing-me-apart-wrong.gif)  |  ![](https://raw.githubusercontent.com/smacke/ffsubsync/master/resources/img/tearing-me-apart-correct.gif)

Helping Development
-------------------
Please consider [supporting Ukraine](https://github.com/vshymanskyy/StandWithUkraine/blob/main/docs/README.md)
rather than donating directly to this project. That said, at the request of
some, you can now help cover my coffee expenses using the Github Sponsors
button at the top, or using the below Paypal Donate button:

[![Donate](https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=XJC5ANLMYECJE)

Install
-------
First, make sure ffmpeg is installed. On MacOS, this looks like:
~~~
brew install ffmpeg
~~~
(Windows users: make sure `ffmpeg` is on your path and can be referenced
from the command line!)

Next, grab the package (compatible with Python >= 3.6):
~~~
pip install ffsubsync
~~~
If you want to live dangerously, you can grab the latest version as follows:
~~~
pip install git+https://github.com/smacke/ffsubsync@latest
~~~

Usage
-----
`ffs`, `subsync` and `ffsubsync` all work as entrypoints:
~~~
ffs video.mp4 -i unsynchronized.srt -o synchronized.srt
~~~

There may be occasions where you have a correctly synchronized srt file in a
language you are unfamiliar with, as well as an unsynchronized srt file in your
native language. In this case, you can use the correctly synchronized srt file
directly as a reference for synchronization, instead of using the video as the
reference:

~~~
ffsubsync reference.srt -i unsynchronized.srt -o synchronized.srt
~~~

`ffsubsync` uses the file extension to decide whether to perform voice activity
detection on the audio or to directly extract speech from an srt file.

Remote URL Support
------------------
`ffsubsync` supports using remote URLs as video references. This allows you to
sync subtitles directly with online video content without downloading the file first:

~~~
ffs "https://example.com/video.mp4" -i unsynchronized.srt -o synchronized.srt
~~~

Supported protocols include:
- `http://` and `https://`
- `rtmp://` (streaming)
- `rtsp://` (streaming)
- `ftp://`

**Note**: Remote URL processing depends on network stability. For large files or
unstable connections, consider downloading the video first for more reliable results.

### Performance Optimization for Remote URLs

For faster processing of remote videos, you can use these options:

~~~
# Extract audio to temp file first (recommended for remote URLs)
ffs "https://example.com/video.mp4" -i sub.srt -o out.srt --extract-audio-first

# Only process first N seconds (useful for long videos)
ffs "https://example.com/video.mp4" -i sub.srt -o out.srt --max-duration-seconds 600

# Combine both for maximum speed
ffs "https://example.com/video.mp4" -i sub.srt -o out.srt --extract-audio-first --max-duration-seconds 600
~~~

**Speed comparison** (for a 2-hour remote video):
| Method | Approximate Time |
|--------|------------------|
| Direct streaming | ~20 minutes |
| `--extract-audio-first` | ~5-8 minutes |
| `--max-duration-seconds 600` | ~3-5 minutes |
| `--multi-segment-sync` | ~2-4 minutes |

### Multi-Segment Sync (Recommended for Long Remote Videos)

For long remote videos, multi-segment sync samples multiple short segments instead of
processing the entire video. This is significantly faster and more robust:

~~~
# Enable multi-segment sync (default: 8 segments × 60 seconds each)
ffs "https://example.com/video.mp4" -i sub.srt -o out.srt --multi-segment-sync

# Customize segment count (more segments = more accurate but slower)
ffs "https://example.com/video.mp4" -i sub.srt -o out.srt --multi-segment-sync --segment-count 10

# Skip intro/outro (first 30s and last 60s) to avoid silent sections
ffs "https://example.com/video.mp4" -i sub.srt -o out.srt --multi-segment-sync --skip-intro-outro

# Adjust parallel workers for segment extraction (default=4)
ffs "https://example.com/video.mp4" -i sub.srt -o out.srt --multi-segment-sync --parallel-workers 6
~~~

**How it works**:
1. Probes video duration
2. Extracts N segments in parallel (default 4 workers) for faster download
3. Samples N segments (default 8) distributed evenly across the video
4. Computes alignment offset for each segment using VAD
5. Returns weighted median offset (filters noise/outliers by score)

**Options**:
| Option | Default | Description |
|--------|---------|-------------|
| `--segment-count` | 8 | Number of 60-second segments to sample |
| `--skip-intro-outro` | off | Skip first 30s and last 60s (intro/credits) |
| `--parallel-workers` | 4 | Parallel workers for segment extraction |

**Benefits**:
- ~80-90% faster than full video processing for 2+ hour videos
- Parallel extraction further speeds up remote URL processing (~60% faster)
- More robust against localized noise (ads, silent sections)
- Automatically falls back to single-segment if video is too short

### Frame Rate and Accuracy

The default frame rate (48000 Hz) provides maximum accuracy. For faster processing
with acceptable accuracy loss, you can lower it:

~~~
# Faster processing with slightly reduced accuracy
ffs video.mp4 -i sub.srt -o out.srt --frame-rate 16000
~~~

| Frame Rate | Speed | Accuracy |
|------------|-------|----------|
| 48000 (default) | Baseline | Highest |
| 16000 | ~3x faster | High (recommended minimum) |
| 8000 | ~6x faster | Medium (may have ±0.1s error) |

Sync Issues
-----------
If the sync fails, the following recourses are available:
- Try to sync assuming identical video / subtitle framerates by passing
  `--no-fix-framerate`;
- Try passing `--gss` to use [golden-section search](https://en.wikipedia.org/wiki/Golden-section_search)
  to find the optimal ratio between video and subtitle framerates (by default,
  only a few common ratios are evaluated);
- Try a value of `--max-offset-seconds` greater than the default of 60, in the
  event that the subtitles are out of sync by more than 60 seconds (empirically
  unlikely in practice, but possible).
- Try `--vad=auditok` since [auditok](https://github.com/amsehili/auditok) can
  sometimes work better in the case of low-quality audio than WebRTC's VAD.
  Auditok does not specifically detect voice, but instead detects all audio;
  this property can yield suboptimal syncing behavior when a proper VAD can
  work well, but can be effective in some cases.

If the sync still fails, consider trying one of the following similar tools:
- [sc0ty/subsync](https://github.com/sc0ty/subsync): does speech-to-text and looks for matching word morphemes
- [kaegi/alass](https://github.com/kaegi/alass): rust-based subtitle synchronizer with a fancy dynamic programming algorithm
- [tympanix/subsync](https://github.com/tympanix/subsync): neural net based approach that optimizes directly for alignment when performing speech detection
- [oseiskar/autosubsync](https://github.com/oseiskar/autosubsync): performs speech detection with bespoke spectrogram + logistic regression
- [pums974/srtsync](https://github.com/pums974/srtsync): similar approach to ffsubsync (WebRTC's VAD + FFT to maximize signal cross correlation)

### Quality Protection (Keep Original on Low Quality)

For short videos or poor audio quality, alignment may fail and produce worse results than
the original subtitles. Use `--skip-sync-on-low-quality` to automatically detect low-quality
alignments and output original subtitles without modification:

~~~
ffs video.mp4 -i sub.srt -o out.srt --skip-sync-on-low-quality
~~~

Quality is determined by three thresholds:

| Option | Default | Description |
|--------|---------|-------------|
| `--min-score` | 0.0 | Minimum alignment score (negative = poor match) |
| `--quality-max-offset-seconds` | 30.0 | Maximum allowed offset in seconds |
| `--max-framerate-deviation` | 0.05 | Maximum framerate scale deviation from 1.0 (5%) |

If any threshold is exceeded, the original subtitles are output unchanged:

~~~
# Custom thresholds
ffs video.mp4 -i sub.srt -o out.srt --skip-sync-on-low-quality \
    --min-score 100 \
    --quality-max-offset-seconds 20 \
    --max-framerate-deviation 0.03
~~~

**When to use**:
- Short videos (< 5 minutes) where alignment data is sparse
- Batch processing where some files may have poor audio
- When original subtitles are already close to correct

### Adaptive Thresholds

For automatic threshold adjustment based on video characteristics, use `--adaptive-thresholds`:

~~~
ffs video.mp4 -i sub.srt -o out.srt --skip-sync-on-low-quality --adaptive-thresholds
~~~

Adaptive thresholds automatically adjust based on:
- **Video duration**: Shorter videos use more relaxed thresholds
- **Speech density**: Low speech density (< 30%) further relaxes thresholds

| Video Duration | Score Multiplier | Offset Multiplier | Framerate Dev Multiplier |
|----------------|------------------|-------------------|--------------------------|
| < 5 min | 0.5× | 0.3× (max 25% of duration) | 1.5× |
| < 30 min | 0.8× | 0.5× (max 25% of duration) | 1.2× |
| >= 30 min | 1.0× | 1.0× | 1.0× |

Speed
-----
`ffsubsync` usually finishes in 20 to 30 seconds, depending on the length of
the video. The most expensive step is actually extraction of raw audio. If you
already have a correctly synchronized "reference" srt file (in which case audio
extraction can be skipped), `ffsubsync` typically runs in less than a second.

How It Works
------------
The synchronization algorithm operates in 3 steps:
1. Discretize both the video file's audio stream and the subtitles into 10ms
   windows.
2. For each 10ms window, determine whether that window contains speech.  This
   is trivial to do for subtitles (we just determine whether any subtitle is
   "on" during each time window); for the audio stream, use an off-the-shelf
   voice activity detector (VAD) like
   the one built into [webrtc](https://webrtc.org/).
3. Now we have two binary strings: one for the subtitles, and one for the
   video.  Try to align these strings by matching 0's with 0's and 1's with
   1's. We score these alignments as (# video 1's matched w/ subtitle 1's) - (#
   video 1's matched with subtitle 0's).

The best-scoring alignment from step 3 determines how to offset the subtitles
in time so that they are properly synced with the video. Because the binary
strings are fairly long (millions of digits for video longer than an hour), the
naive O(n^2) strategy for scoring all alignments is unacceptable. Instead, we
use the fact that "scoring all alignments" is a convolution operation and can
be implemented with the Fast Fourier Transform (FFT), bringing the complexity
down to O(n log n).

Limitations
-----------
In most cases, inconsistencies between video and subtitles occur when starting
or ending segments present in video are not present in subtitles, or vice versa.
This can occur, for example, when a TV episode recap in the subtitles was pruned
from video. FFsubsync typically works well in these cases, and in my experience
this covers >95% of use cases. Handling breaks and splits outside of the beginning
and ending segments is left to future work (see below).

Future Work
-----------
Besides general stability and usability improvements, one line
of work aims to extend the synchronization algorithm to handle splits
/ breaks in the middle of video not present in subtitles (or vice versa).
Developing a robust solution will take some time (assuming one is possible).
See [#10](https://github.com/smacke/ffsubsync/issues/10) for more details.

History
-------
The implementation for this project was started during HackIllinois 2019, for
which it received an **_Honorable Mention_** (ranked in the top 5 projects,
excluding projects that won company-specific prizes).

Credits
-------
This project would not be possible without the following libraries:
- [ffmpeg](https://www.ffmpeg.org/) and the [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) wrapper, for extracting raw audio from video
- VAD from [webrtc](https://webrtc.org/) and the [py-webrtcvad](https://github.com/wiseman/py-webrtcvad) wrapper, for speech detection
- [srt](https://pypi.org/project/srt/) for operating on [SRT files](https://en.wikipedia.org/wiki/SubRip#SubRip_text_file_format)
- [numpy](http://www.numpy.org/) and, indirectly, [FFTPACK](https://www.netlib.org/fftpack/), which powers the FFT-based algorithm for fast scoring of alignments between subtitles (or subtitles and video)
- Other excellent Python libraries like [argparse](https://docs.python.org/3/library/argparse.html), [rich](https://github.com/willmcgugan/rich), and [tqdm](https://tqdm.github.io/), not related to the core functionality, but which enable much better experiences for developers and users.

# License
Code in this project is [MIT licensed](https://opensource.org/licenses/MIT).
