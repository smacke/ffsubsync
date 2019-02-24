# subalign
Automagically synchronize subtitles with video, aligning them to the correct starting point.

Turn this:                       |  Into this:
:-------------------------------:|:-------------------------:
![](tearing-me-apart-wrong.gif)  |  ![](tearing-me-apart-correct.gif)

# install
~~~
pip install git+https://github.com/smacke/subalign
~~~

# usage
~~~
subalign video.mp4 -i unsynchronized.srt > synchronized.srt
~~~

or

~~~
subalign video.mp4 -i unsynchronized.srt -o synchronized.srt
~~~

Although it can usually work if all you have is the video file, it will be faster (and potentially more accurate) if you have a correctly synchronized "reference" srt file, in which case you can do the following:

~~~
subalign reference.srt -i unsynchronized.srt -o synchronized.srt
~~~

Whether to perform voice activity detection on the audio or to directly extract speech from an srt file is determined from the file extension.

# credits
This script makes use of webrct-vad and ffmpeg / ffmpeg-python for all the audio / speech extraction.

# license
TODO figure this out.
