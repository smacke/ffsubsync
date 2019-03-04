#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


def read_file(fname):
    with open(fname, 'r') as f:
        return f.read()


history = read_file('HISTORY.rst')

requirements = read_file('requirements.txt').strip().split()

setup(
    name='subsync',
    version='0.1.6',
    author='Stephen Macke',
    author_email='stephen.macke@gmail.com',
    description='Language-agnostic synchronization of subtitles with video via speech detection.',
    long_description=read_file('README.md'),
    url='https://github.com/smacke/subsync',
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': ['subsync = subsync:main'],
    },
    license='MIT',
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)

# python setup.py sdist bdist_wheel --universal
# twine upload dist/*
