# -*- coding: utf-8 -*-
.PHONY: clean dist deploy check test tests deps devdeps

clean:
	rm -rf dist/ *.egg-info/

dist: clean
	python setup.py sdist

deploy: dist
	twine upload dist/*

check:
	INTEGRATION=1 pytest

test: check
tests: check

deps:
	pip install -r requirements.txt

devdeps:
	pip install -e .
	pip install -r requirements-dev.txt
