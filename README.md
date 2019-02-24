# subalign
Automagically synchronize subtitles with video, aligning them to the correct starting point.

Turn this:
<img style="float: right;" src="tearing-me-apart-wrong.gif" />

Into this:
![](tearing-me-apart-correct.gif)

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
