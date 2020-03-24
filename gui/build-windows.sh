#!/usr/bin/env bash
#docker run -v "$(pwd):/src/" -v "$(pwd)/..:/subsync/" --entrypoint /bin/sh cdrx/pyinstaller-windows:python3-32bit -c "pip install -e /subsync && pip install -r /src/requirements.txt && pyinstaller --debug all --clean -y --dist ./dist/windows --workpath /tmp *.spec"
docker run -v "$(pwd):/src/" -v "$(pwd)/..:/subsync/" --entrypoint /bin/sh cdrx/pyinstaller-windows:python3-32bit -c "pip install -e /subsync && /entrypoint.sh"
