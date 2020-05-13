FFsubsync
=======

[![CI Status](https://github.com/smacke/ffsubsync/workflows/master/badge.svg)](https://github.com/smacke/ffsubsync/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-maroon.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/ffsubsync.svg)](https://pypi.org/project/ffsubsync)
[![Documentation Status](https://readthedocs.org/projects/ffsubsync/badge/?version=latest)](https://ffsubsync.readthedocs.io/en/latest/?badge=latest)
[![PyPI Version](https://img.shields.io/pypi/v/ffsubsync.svg)](https://pypi.org/project/ffsubsync)


Language-agnostic automatic synchronization of subtitles with video, so that
subtitles are aligned to the correct starting point within the video.

Turn this:                       |  Into this:
:-------------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/smacke/ffsubsync/master/tearing-me-apart-wrong.gif)  |  ![](https://raw.githubusercontent.com/smacke/ffsubsync/master/tearing-me-apart-correct.gif)

Install
-------
First, make sure ffmpeg is installed. On MacOS, this looks like:
~~~
brew install ffmpeg
~~~
Next, grab the script. It should work with both Python 2 and Python 3:
~~~
pip install ffsubsync
~~~
If you want to live dangerously, you can grab the latest version as follows:
~~~
pip install git+https://github.com/smacke/ffsubsync@latest
~~~

Usage
-----
Both `subsync` and `ffsubsync` work as entrypoints:
~~~
ffsubsync video.mp4 -i unsynchronized.srt > synchronized.srt
~~~

or

~~~
ffsubsync video.mp4 -i unsynchronized.srt -o synchronized.srt
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

Speed
-----
`ffsubsync` usually finishes in 20 to 30 seconds, depending on the length of the
video. The most expensive step is actually extraction of raw audio. If you
already have a correctly synchronized "reference" srt file (in which case audio
extraction can be skipped), `ffsubsync` typically runs in less than a second.

How It Works
------------
The synchronization algorithm operates in 3 steps:
1. Discretize video and subtitles by time into 10ms windows.
2. For each 10ms window, determine whether that window contains speech.  This
   is trivial to do for subtitles (we just determine whether any subtitle is
   "on" during each time window); for video, use an off-the-shelf voice
   activity detector (VAD) like
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
- [sklearn](https://scikit-learn.org/) for its data pipeline API
- Other excellent Python libraries like [argparse](https://docs.python.org/3/library/argparse.html) and [tqdm](https://tqdm.github.io/), not related to the core functionality, but which enable much better experiences for developers and users.

# License
Code in this project is [MIT licensed](https://opensource.org/licenses/MIT).
