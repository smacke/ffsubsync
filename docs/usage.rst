Usage
=====

At its core, ffsubsync takes a **reference** (something correctly timed) and an
**input subtitle** (something mistimed), and writes a new subtitle shifted to
match the reference. ``ffs``, ``subsync``, and ``ffsubsync`` are interchangeable
entry points.

Sync against a video
--------------------

The most common case: you have a video and an out-of-sync subtitle.

.. code-block:: console

   $ ffs video.mp4 -i unsynchronized.srt -o synchronized.srt

ffsubsync extracts the video's audio, runs voice-activity detection to work out
when someone is speaking, and finds the offset that best lines the subtitle's
"on" intervals up with the detected speech.

Sync against another subtitle
-----------------------------

Sometimes you have a subtitle that is *already* correctly synced — perhaps in a
language you don't read — plus an out-of-sync subtitle in your own language. You
can use the correct subtitle directly as the reference:

.. code-block:: console

   $ ffs reference.srt -i unsynchronized.srt -o synchronized.srt

ffsubsync decides what to do based on the reference's file extension: a subtitle
extension (``.srt``, ``.ass``, ``.ssa``, ``.sub``) skips audio extraction
entirely and derives the speech signal directly from the reference's subtitle
timings. Because there is no audio to decode, this runs in **under a second**.
See :doc:`reference_types` for the full list of things that can serve as a
reference.

Let ffsubsync find the input for you
------------------------------------

If you omit ``-i``, ffsubsync looks for subtitle files sitting next to the
reference that share its name, and syncs each of them:

.. code-block:: console

   $ ffs video.mp4

For a reference named ``video.mp4`` this picks up siblings like ``video.srt`` and
``video.en.srt`` from the same directory, and writes the synced result for each
to a ``<name>.synced.srt`` alongside it (e.g. ``video.synced.srt``), leaving the
originals untouched. Previously-produced ``*.synced.srt`` files are skipped, so
re-running is safe. Pass ``--overwrite-input`` to rewrite the detected files in
place instead of producing ``.synced.srt`` copies.

You can pass multiple references at once. Shell globs are expanded into separate
references, and each reference independently auto-detects its sibling subtitles:

.. code-block:: console

   $ ffs --overwrite-input *.mkv

Explicit ``-i`` and ``-o`` paths are not accepted with multiple references
because they would be ambiguous, and batch mode does not consume subtitles from
stdin. References without matching external subtitles are skipped. A failure for
one reference does not prevent the remaining references from being processed;
the command exits non-zero if any processed reference fails.

Sibling auto-detection is local-only. In single-reference mode, it is also
skipped when subtitles are piped in on stdin.

Reading and writing standard streams
------------------------------------

``-i`` defaults to stdin and ``-o`` defaults to stdout, so ffsubsync composes
cleanly in a pipeline:

.. code-block:: console

   $ cat unsynchronized.srt | ffs video.mp4 > synchronized.srt

Progress and log messages are written to stderr, so they won't corrupt piped
subtitle output.

Remote references
-----------------

The reference can be a remote URL instead of a local file. Anything ffmpeg can
read works as a video/audio reference, and remote subtitle files work too:

.. code-block:: console

   $ ffs "https://example.com/video.mp4" -i unsynchronized.srt -o synchronized.srt
   $ ffs "https://example.com/reference.srt" -i unsynchronized.srt -o synchronized.srt

Supported protocols are ``http(s)://``, ``rtmp://``, ``rtsp://``, and ``ftp://``.
Processing streams the reference over the network, so reliability depends on the
connection. For large or flaky sources there are several options —
``--max-duration-seconds``, ``--extract-audio-first``, and
``--multi-segment-sync`` — described under :ref:`long-and-remote-references`.

What you get back
-----------------

On success, ffsubsync writes the shifted subtitle and reports the offset (and
framerate scale factor, if it corrected one) on stderr. If you'd rather drive
ffsubsync from Python and inspect these values programmatically, see
:doc:`library`.
