#!/usr/bin/env bash
docker run -v "$(pwd):/src/" -v "$(pwd)/..:/subsync/" --entrypoint /bin/sh cdrx/pyinstaller-windows -c "pip install -e /subsync && pip install -r /src/requirements.txt && /entrypoint.sh"
