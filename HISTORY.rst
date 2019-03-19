History
=======

0.1.0 (2019-02-24)
------------------
* Support srt format;
* Support using srt as reference;
* Support using video as reference (via ffmpeg);
* Support writing to stdout or file (read from stdin not yet supported; can only read from file);

0.1.6 (2019-03-04)
------------------
* Misc bugfixes;
* Proper logging;
* Proper version handling;

0.1.7 (2019-03-05)
------------------
* Add Chinese to the list of encodings that can be inferred;
* Make srt parsing more robust;

0.2.0 (2019-03-06)
------------------
* Get rid of auditok (GPLv3, was hurting alignment algorithm);
* Change to alignment algo: don't penalize matching video non-speech with subtitle speech;

0.2.1 (2019-03-07)
------------------
* Developer note: change progress-only to vlc-mode and remove from help docs;

0.2.2 (2019-03-08)
------------------
* Allow reading input srt from stdin;
* Allow specifying encodings for reference, input, and output srt;
* Use the same encoding for both input srt and output srt by default;
* Developer note: using sklearn-style data pipelines now;

0.2.3 (2019-03-08)
------------------
* Minor change to subtitle speech extraction;

0.2.4 (2019-03-19)
------------------
* Add six to requirements.txt;
* Set default encoding to utf8 to ensure non ascii filenames handled properly;
