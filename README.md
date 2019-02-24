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

# VLC Integration
To demonstrate how one might use subsync seamlessly with real video software,
we developed a prototype integration into the popular [VLC](https://www.videolan.org/vlc/index.html)
media player, which was demoed during the Hackillinois 2019 project expo. The resulting patch
can be found in the file [subsync-vlc.patch](https://github.com/smacke/subsync/raw/master/subsync-vlc.patch).
Here are instructions for how to use it.

1. First clone the 3.0 maintenance branch of VLC and checkout 3.0.6:
~~~
git clone git://git.videolan.org/vlc/vlc-3.0.git
cd vlc-3.0
git checkout 3.0.6
~~~
2. Next, apply the patch:
~~~
wget https://github.com/smacke/subsync/raw/master/subsync-vlc.patch
git apply subsync-vlc.patch
~~~
3. Follow the normal instructions on the
[VideoLAN wiki](https://wiki.videolan.org/VLC_Developers_Corner/)
for building VLC from source. *Warning: this is not easy.*

You should now be able to autosynchronize subtitles using the hotkey `Ctrl+Shift+S`
(only enabled while subtitles are present).

# Future Work
The prototype VLC patch is very experimental -- it developed under pressure
and just barely works. The clear next step for this project is a more robust
integration with VLC, either directly in the VLC core, or as a plugin.

# Credits
This project would not be possible without the following libraries:
- [ffmpeg](https://www.ffmpeg.org/) and the [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) wrapper, for extracting raw audio from video
- VAD from [webrtc](https://webrtc.org/) and the [py-webrtcvad](https://github.com/wiseman/py-webrtcvad) wrapper, for speech detection
- [auditok](https://pypi.org/project/auditok/), for backup audio detection if webrtcvad misbehaves
- [srt](https://pypi.org/project/srt/) for operating on [SRT files](https://en.wikipedia.org/wiki/SubRip#SubRip_text_file_format)
- [numpy](http://www.numpy.org/) and, indirectly, [FFTPACK](https://www.netlib.org/fftpack/), which powers the FFT-based algorithm for fast scoring of alignments between subtitles (or subtitles and video).
- Other excellent Python libraries like [argparse](https://docs.python.org/3/library/argparse.html) and [tqdm](https://tqdm.github.io/), not related to the core functionality, but which enable much better experiences for developers and users.

# License
Code in this project is [MIT Licensed](https://opensource.org/licenses/MIT).
