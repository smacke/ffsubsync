#!/usr/bin/env bash
python3 -m PyInstaller --clean -y --dist ./dist/macos build.spec
# ref: https://github.com/chriskiehl/Gooey/issues/259#issuecomment-522432026
mkdir -p ./dist/macos/Contents
