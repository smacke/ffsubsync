#!/usr/bin/env bash
rm -r dist/
python setup.py sdist
twine upload dist/*
