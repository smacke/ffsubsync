#!/usr/bin/env bash

set -Eeuxo pipefail

BASE=.
DIST="$BASE/dist"
BUILD="$BASE/build/dmg"
VERSION=$(python3 -c "from subsync.version import __version__; print(__version__)")
APP="Subsync.app"
TARGET="$DIST/subsync-${VERSION}-mac-x86_64.dmg"

test -e "$BUILD" && rm -rf "$BUILD"
test -e "$TARGET" && rm -f "$TARGET"
mkdir -p "$BUILD"
cp -r "$DIST/$APP" "$BUILD"

create-dmg \
    --volname "subsync installer" \
    `#--volicon "icon.icns"` \
    --window-pos 300 200 \
    --window-size 700 500 \
    --icon-size 150 \
    --icon "$APP" 200 200 \
    --hide-extension "$APP" \
    --app-drop-link 450 200 \
    --no-internet-enable \
    "$TARGET" "$BUILD"
