#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
The setup script for the calibraxis package.

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2016-04-08

'''

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import re
from codecs import open

from setuptools import setup, find_packages


if sys.argv[-1] == 'publish':
    os.system('python setup.py register')
    os.system('python setup.py sdist upload')
    os.system('python setup.py bdist_wheel upload')
    sys.exit()


with open('calibraxis.py', 'r') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)


def read(f):
    return open(f, encoding='utf-8').read()


setup(
    name='calibraxis',
    version=version,
    author='Henrik Blidh',
    author_email='henrik.blidh@nedobmkull.com',
    url='https://github.com/hbldh/calibraxis',
    description='Autocalibration method for accelerometers, implemented in Python.',
    long_description=read('README.rst'),
    license='MIT',
    platforms=['Linux'],
    keywords=['Calibration', 'Accelerometers'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    py_modules=['calibraxis'],
    test_suite="tests",
    zip_safe=False,
    include_package_data=True,
    platforms='any',
    install_requires=[
        'numpy>=1.9.0',
        'six>=1.9.0'
    ],
    setup_requires=['pytest-runner', ],
    tests_require=['pytest', ],
    ext_modules=[],
    entry_points={}
)
