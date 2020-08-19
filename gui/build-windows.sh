#!/usr/bin/env bash
nbits=${1:-64}
tag="python3"
if [[ "$nbits" == 32 ]]; then
    tag="${tag}-32bit"
fi
docker run -v "$(pwd):/src/" -v "$(pwd)/..:/ffsubsync/" --entrypoint /bin/sh "cdrx/pyinstaller-windows:${tag}" -c "pip install -e /ffsubsync && /ffsubsync/gui/entrypoint-windows.sh"
rm -r "./dist/win${nbits}"
mv ./dist/windows "./dist/win${nbits}"
