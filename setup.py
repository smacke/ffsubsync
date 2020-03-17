#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages


def read_file(fname):
    with open(fname, 'r') as f:
        return f.read()


history = read_file('HISTORY.rst')
requirements = read_file('requirements.txt').strip().split()
pkg_name = 'ffsubsync'
exec(read_file(os.path.join(pkg_name, 'version.py')))
setup(
    name=pkg_name,
    version=__version__,  # noqa
    author='Stephen Macke',
    author_email='stephen.macke@gmail.com',
    description='Language-agnostic synchronization of subtitles with video.',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/smacke/ffsubsync',
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'subsync = ffsubsync:main'
            'ffsubsync = ffsubsync:main'
        ],
    },
    license='MIT',
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
    ],
)

# python setup.py sdist
# twine upload dist/*
