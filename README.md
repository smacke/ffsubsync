# subalign
Automagically synchronize subtitles with video, aligning them to the correct starting point.

<p style="width:100px;display:inline-block">Turn this:</p>
<img src="tearing-me-apart-wrong.gif" />

<br />
~~~
======> infer that subtitles should be moved 15min forward ======>
~~~

<p style="width:100px;display:inline-block">Into this:</p>
<img src="tearing-me-apart-correct.gif" />

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

Whether to do voice activity detection on the audo versus directly extract speech from an srt file is determined from the extension.

# credits
This script makes use of webrct-vad and ffmpeg / ffmpeg-python for all the audio / speech extraction.

# license
TODO figure this out.
