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

0.3.5 (2020-05-08)
------------------
* Fix corner case bug that occurred when multiple sync attempts were scored the same;

0.3.7 (2020-05-11)
------------------
* Fix PyPI issues;

0.4.0 (2020-06-02)
------------------
* Remove dependency on scikit-learn;
* Implement PyInstaller / Gooey build process for graphical application on MacOS and Windows;

0.4.1 (2020-06-06)
------------------
* Add --reference-stream option for selecting the stream / track from the video reference to use for speech detection;

0.4.2 (2020-06-06)
------------------
* Fix Python 2 compatibility bug;

0.4.3 (2020-06-07)
------------------
* Fix regression where stdout not used for default output;
* Add ability to specify path to ffmpeg / ffprobe binaries;
* Add ability to overwrite the input / unsynced srt with the --overwrite-input flag;

0.4.4 (2020-06-08)
------------------
* Use rich formatting for Python >= 3.6;
* Use versioneer to manage versions;

0.4.5 (2020-06-09)
------------------
* Allow MicroDVD input format;
* Use output extension to determine output format;

0.4.6 (2020-06-10)
------------------
* Bugfix for writing subs to stdout;

0.4.7 (2020-09-05)
------------------
* Misc bugfixes and stability improvements;

0.4.8 (2020-09-22)
------------------
* Use webrtcvad-wheels on Windows to eliminate dependency on compiler;

0.4.9 (2020-10-11)
------------------
* Make default max offset seconds 60 and enforce during alignment as opposed to throwing away alignments with > max_offset_seconds;
* Add experimental section for using golden section search to find framerate ratio;
* Restore ability to read stdin and write stdout after buggy permissions check;
* Exceptions that occur during syncing were mistakenly suppressed; this is now fixed;

0.4.10 (2021-01-18)
-------------------
* Lots of improvements from PRs submitted by @alucryd (big thanks!):
    * Retain ASS styles;
    * Support syncing several subs against the same ref via --overwrite-input flag;
    * Add --apply-offset-seconds postprocess option to shift alignment by prespecified amount;
* Filter out metadata in subtitles when extracting speech;
* Add experimental --golden-section-search over framerate ratio (off by default);
* Try to improve sync by inferring framerate ratio based on relative duration of synced vs unsynced;

0.4.11 (2021-01-29)
-------------------
* Misc sync improvements:
    * Have webrtcvad use '0' as the non speech label instead of 0.5;
    * Allow the vad non speech label to be specified via the --non-speech-label command line parameter;
    * Don't try to infer framerate ratio based on length between first and last speech frames for non-subtitle speech detection;

0.4.12 (2021-04-13)
-------------------
* Pin auditok to 0.1.5 to avoid API-breaking change

0.4.13 (2021-05-10)
-------------------
* Support SSA embedded fonts using new pysubs2 'opaque_fonts' metadata;
* Set min required pysubs2 version to 1.2.0 to ensure the aforementioned functionality is available;

0.4.14 (2021-05-10)
-------------------
* Hotfix for pysubs2 on Python 3.6;

0.4.15 (2021-05-25)
-------------------
* Make typing_extensions a requirement

0.4.16 (2021-07-22)
-------------------
* Fix a couple of validation bugs that prevented certain uncommon command line options from use;

0.4.17 (2021-10-02)
-------------------
* Add --suppress-output-if-offset-less-than arg to suppress output for small syncs;

0.4.17 (2021-10-03)
-------------------
* Don't remove log file if --log-dir-path explicitly requested;
