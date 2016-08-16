#!/usr/bin/env python

"""
setup.py for pyReef
"""
from numpy.distutils.core import setup, Extension

ext_modules = []

setup(
    name="pyReef",
    version="0.1",
    author="Tristan Salles",
    author_email="",
    description=("Carbonate Platform Model"),
    long_description=open('README.md').read(),
    classifiers=[
        "Development Status :: 1 - Alpha",
    ],
    packages=['pyReef', 'pyReef.libUtils'],
    ext_package='pyReef',
    ext_modules=ext_modules,
    scripts=[],
)
