# subsync
Automagically synchronize subtitles with video, aligning them to the correct starting point.

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
This script makes use of webrct-vad and ffmpeg / ffmpeg-python for all the audio / speech extraction.

# License
TODO figure this out.
