== Note on platform-specific PyInstaller version in requirements.txt ==

PyInstaller>=3.6 introduces a webrtcvad hook that seems to not play nicely
with the webrtcvad-wheels package. This package contains prebuilt wheels
and is needed for Windows (unless I can get a working C compiler in my
Windows build environment, which is doubtful). For MacOS this isn't a
problem since I can use the vanilla webrtcvad package and leverage the
preexisting hook in PyInstaller>=3.6, but for Windows I need to use the
old version of PyInstaller without the hook and introduce my own (in the
'hooks' directory).

== Note on Scikit-Learn ==
There is some DLL that wasn't getting bundled in the Windows PyInstaller
build and causing the built exe to complain. My solution was to remove
the dependency and include a shim for the Pipeline / Transformer fuctionality.
