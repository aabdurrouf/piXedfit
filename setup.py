#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
#import glob
#import subprocess
try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

#githash = subprocess.check_output(["git", "log", "--format=%h"], universal_newlines=True).split('\n')[0]
vers = "0.1.1"
#githash = ""
#with open('prospect/_version.py', "w") as f:
#    f.write('__version__ = "{}"\n'.format(vers))
#    f.write('__githash__ = "{}"\n'.format(githash))

setup(
    name="pixedfit",
    version=vers,
    project_urls={"Source repo": "https://github.com/aabdurrouf/piXedfit",
                  "Documentation": "https://pixedfit.readthedocs.io"},
    author="Abdurrouf",
    author_email="fabdurr1@jhu.edu",
    classifiers=["Development Status :: 4 - Beta",
                 "Intended Audience :: Science/Research",
                 "Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Natural Language :: English",
                 "Topic :: Scientific/Engineering :: Astronomy"],
    packages=["piXedfit",
              "piXedfit.piXedfit_analysis",
              "piXedfit.piXedfit_bin",
              "piXedfit.piXedfit_fitting",
              "piXedfit.piXedfit_images",
              "piXedfit.piXedfit_model",
              "piXedfit.piXedfit_spectrophotometric",
              "piXedfit.utils"],
    python_requires=">=3.7, <4",
    license="MIT",
    description="A python package specially designed for SED fitting of resolved sources",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    package_data={"": ["README.md", "LICENSE"]},
    #scripts=glob.glob("scripts/*.py"),
    include_package_data=True,
    install_requires=["numpy", "h5py", "scipy", "astropy", "photutils==1.1.0", "matplotlib", "reproject", 
                      "sep", "emcee", "schwimmbad", "astroquery", "pyvo"],
)
