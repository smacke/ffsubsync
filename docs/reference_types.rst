Reference types
===============

The **reference** is whatever ffsubsync treats as the ground truth for timing.
ffsubsync inspects the reference — mostly its file extension — and picks one of
several strategies for turning it into a speech signal to align against.
Understanding these paths helps you choose the fastest and most accurate option
for what you have on hand.

Video or audio (voice-activity detection)
-----------------------------------------

When the reference is a media file, ffsubsync uses ffmpeg to extract the audio
and then runs a **voice-activity detector (VAD)** to label each 10 ms window as
speech or silence. This is the most general path — it works for any video with a
dialogue track — but also the most expensive, since audio extraction dominates
the runtime.

Which detector runs, and how to tune it for difficult audio, is covered under
:ref:`vad-backends`. The audio is extracted at a sample rate controlled by
``--frame-rate`` (default 48000; this is the *audio* sample rate used for VAD,
not the video's frames per second).

Embedded subtitles first (``subs_then_*``)
------------------------------------------

Many video containers (especially MKV) carry one or more **embedded text
subtitle** streams. Those are already a perfect speech signal — far cheaper and
often more accurate than running a VAD over the audio.

The default detector, ``subs_then_webrtc``, exploits this: it first tries to use
an embedded text-subtitle stream from the reference, and only falls back to the
WebRTC audio VAD if no usable embedded subtitles are found. The ``subs_then_*``
family (``subs_then_webrtc``, ``subs_then_auditok``, ``subs_then_silero``) all
behave this way, differing only in which audio VAD they fall back to. Use a bare
detector name (e.g. ``--vad webrtc``) to skip the embedded-subtitle shortcut and
force audio detection.

Subtitle file
-------------

If the reference itself is a subtitle file — extension ``.srt``, ``.ass``,
``.ssa``, or ``.sub`` — ffsubsync derives the speech signal straight from the
reference's on/off subtitle timings. No audio is extracted, so this is the
fastest path (typically under a second). This is the "sync against an
already-correct subtitle" workflow from :doc:`usage`.

When the reference is a subtitle file you can also control its text encoding with
``--reference-encoding`` (it defaults to auto-detection, just like input
subtitles — see :doc:`encoding`), and merge the reference into the output with
``--merge-with-reference``.

Image-based subtitles (PGS, VobSub, DVB)
----------------------------------------

Blu-ray rips often ship subtitles as **PGS** (Presentation Graphic Stream)
image-based tracks rather than text, DVD rips as **VobSub**, and broadcast
recordings as **DVB** subtitles. ffsubsync can use any of these as the sync
reference without any OCR, deriving speech timing from when each subtitle image
is displayed:

.. code-block:: console

   $ ffs ref.mkv -i in.srt -o out.srt --pgs-ref-stream

Passing ``--pgs-ref-stream`` with no value auto-detects an image-based track.
Commentary tracks and forced tracks (which usually carry only signs and
foreign-language lines) are passed over while there is another; among the rest,
the track with the most subtitles is used when the file records it (as mkvmerge
does), and otherwise the first. To pick a specific track, give it a stream
specifier (the leading ``0:`` is optional):

.. code-block:: console

   $ ffs ref.mkv -i in.srt -o out.srt --pgs-ref-stream s:2

Whisper transcription (``--whisper-weights``)
---------------------------------------------

When a video has **no subtitles at all** — not embedded text, not image-based — the only
reference is the audio. Instead of a plain VAD, recent ffmpeg builds (>= 8.0,
compiled with ``--enable-whisper``) can transcribe the audio with
`whisper.cpp <https://github.com/ggml-org/whisper.cpp>`_ in a single pass.
ffsubsync uses that transcript's cue timings as the reference signal — often a
sharper speech/silence signal than energy-based VAD, at the cost of running a
speech-recognition model.

Point ``--whisper-weights`` at a whisper.cpp ggml model file:

.. code-block:: console

   $ ffs video.mp4 -i in.srt -o out.srt \
       --whisper-weights ~/whisper.cpp/models/ggml-base.en.bin

ffsubsync smooths over ffmpeg's rough edges here:

- **``~`` is expanded** for the model path (ffmpeg itself won't do this).
- **The language is inferred**: an English-only model (named ``*.en.bin``) uses
  ``en``; otherwise whisper auto-detects. Override with ``--language`` (e.g.
  ``--language es``, or ``--language auto`` to force detection). You don't need
  to remember ffmpeg's filter-string syntax.
- **A warning is shown** if the reference already contains embedded subtitle
  streams, since those are usually a better (and much cheaper) reference — see
  ``subs_then_*`` above.
- **Clear errors** are raised if the weights file is missing or if your ffmpeg
  was not built with the whisper filter.

To tune the underlying ffmpeg whisper filter, pass extra ``key=value`` options
with ``--whisper-args`` (for example ``--whisper-args queue=12`` to enlarge the
audio window — larger values give more accurate timings but use more CPU). The
``model``, ``format``, and ``destination`` options are managed by ffsubsync and
cannot be overridden.

whisper's filter also supports an optional VAD model that fragments the audio
before transcription. To enable it, **reuse the** ``--vad`` **flag** with a path
to a ggml VAD model (e.g. ``--vad ~/whisper.cpp/models/ggml-silero-v5.1.2.bin``);
in transcription mode the named VAD detectors do not apply, so ``--vad`` carries
that path instead.

Serialized speech (``.npy`` / ``.npz``)
---------------------------------------

If you pass a ``.npy`` or ``.npz`` file as the reference, ffsubsync loads a
previously-serialized speech signal instead of computing one. You produce such a
file with ``--serialize-speech`` (see :doc:`advanced`). This is handy when you
want to sync several subtitle files against the same video: extract the speech
signal once, then reuse it repeatedly without re-decoding the audio.

Selecting a stream from the reference
-------------------------------------

A video file can contain several audio or subtitle tracks. Use
``--reference-stream`` to choose which one to use, formatted according to ffmpeg
conventions:

.. code-block:: console

   $ ffs ref.mkv -i in.srt -o out.srt --reference-stream s:2

For example, ``0:s:0`` uses the first subtitle track and ``0:a:3`` uses the
fourth audio track; you may drop the leading ``0:`` and write ``s:0`` or ``a:3``.

Offset-only mode (no reference)
-------------------------------

Finally, ffsubsync doesn't strictly need a reference at all. If you already know
the correction you want, ``--apply-offset-seconds`` shifts every subtitle by a
fixed amount with no alignment step:

.. code-block:: console

   $ ffs -i in.srt -o out.srt --apply-offset-seconds 3.5

This is covered further in :doc:`advanced`.
