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

0.2.5 (2019-05-14)
------------------
* Clamp subtitles to maximum duration (default 10);

0.2.6 (2019-05-15)
------------------
* Fix argument parsing regression;

0.2.7 (2019-05-28)
------------------
* Add utf-16 to list of encodings to try for inference purposes;

0.2.8 (2019-06-15)
------------------
* Allow user to specify start time (in seconds) for processing;

0.2.9 (2019-09-22)
------------------
* Quck and dirty fix to properly handle timestamp ms fields with >3 digits;

0.2.10 (2019-09-22)
------------------
* Specify utf-8 encoding at top of file for backcompat with Python2;

0.2.11 (2019-10-06)
------------------
* Quick and dirty fix to recover without progress info if `ffmpeg.probe` raises;

0.2.12 (2019-10-06)
------------------
* Clear O_NONBLOCK flag on stdout stream in case it is set;

0.2.14 (2019-10-07)
------------------
* Bump min required scikit-learn to 0.20.4;

0.2.15 (2019-10-11)
------------------
* Revert changes from 0.2.12 (caused regression on Windows);

0.2.16 (2019-12-04)
------------------
* Revert changes from 0.2.9 now that srt parses weird timestamps robustly;

0.2.17 (2019-12-21)
------------------
* Try to correct for framerate differences by picking best framerate ratio;

0.3.0 (2020-03-11)
------------------
* Better detection of text file encodings;
* ASS / SSA functionality (but currently untested);
* Allow serialize speech with --serialize-speech flag;
* Convenient --make-test-case flag to create test cases when filing sync-related bugs;
* Use utf-8 as default output encoding (instead of using same encoding as input);
* More robust test framework (integration tests!);

0.3.1 (2020-03-12)
------------------
* Fix bug when handling ass/ssa input, this format should work now;

0.3.2 (2020-03-13)
------------------
* Add ability to merge synced and reference subs into bilingual subs when reference is srt;

0.3.3 (2020-03-15)
------------------
* Hotfix for test archive creation bug;

0.3.4 (2020-03-20)
------------------
* Attempt speech extraction from subtitle tracks embedded in video first before using VAD;
