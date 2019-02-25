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
    version='0.1.1',
    author='Stephen Macke',
    author_email='stephen.macke@gmail.com',
    description='Synchronize subtitles with video with speech extraction, even across different languages.',
    long_description=read_file('README.md'),
    url='https://github.com/smacke/subsync',
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': ['subsync = subsync:main'],
    },
    # license='Apache License 2.0', # TODO: does this have to be (L)GPL because of libs?
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
    ],
)

# python setup.py sdist bdist_wheel --universal
# twine upload dist/*
