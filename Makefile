# -*- coding: utf-8 -*-
.PHONY: clean build bump deploy check test tests deps devdeps

clean:
	rm -rf dist/ build/ *.egg-info/

build: clean
	python setup.py sdist bdist_wheel --universal

bump:
	./scripts/bump-version.py

deploy: build
	./scripts/deploy.sh

check:
	INTEGRATION=1 pytest

test: check
tests: check

deps:
	pip install -r requirements.txt

devdeps:
	pip install -e .
	pip install -r requirements-dev.txt
