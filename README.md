# subsync
Automagically synchronize subtitles with video, aligning them to the correct starting point.

**_This is my submission for Hackillinois 2019._**

Turn this:                       |  Into this:
:-------------------------------:|:-------------------------:
![](tearing-me-apart-wrong.gif)  |  ![](tearing-me-apart-correct.gif)

# Install
First, make sure ffmpeg is installed. On MacOS, this looks like:
~~~
brew install ffmpeg
~~~
Next, grab the script. It should work with both Python2 and Python3:
~~~
pip install git+https://github.com/smacke/subsync
~~~

# Usage
~~~
subsync video.mp4 -i unsynchronized.srt > synchronized.srt
~~~

or

~~~
subsync video.mp4 -i unsynchronized.srt -o synchronized.srt
~~~

Although it can usually work if all you have is the video file, it will be faster (and potentially more accurate) if you have a correctly synchronized "reference" srt file, in which case you can do the following:

~~~
subsync reference.srt -i unsynchronized.srt -o synchronized.srt
~~~

Whether to perform voice activity detection on the audio or to directly extract speech from an srt file is determined from the file extension.

# Credits
This project would not be possible without the following libraries:
- [ffmpeg](https://www.ffmpeg.org/) and the [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) wrapper, for extracting raw audio from video
- VAD from [webrtc](https://webrtc.org/) and the [py-webrtcvad](https://github.com/wiseman/py-webrtcvad) wrapper, for speech detection
- [auditok](https://pypi.org/project/auditok/), for backup audio detection if webrtcvad misbehaves
- [srt](https://pypi.org/project/srt/) for operating on [SRT files](https://en.wikipedia.org/wiki/SubRip#SubRip_text_file_format)
- [numpy](http://www.numpy.org/) and, indirectly, [FFTPACK](https://www.netlib.org/fftpack/), which powers the FFT-based algorithm for fast scoring of alignments between subtitles (or subtitles and video).
- Other excellent Python libraries like [argparse](https://docs.python.org/3/library/argparse.html) and [tqdm](https://tqdm.github.io/), not related to the core functionality, but which enable much better experiences for developers and users.

# License
TODO figure this out.
