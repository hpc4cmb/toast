# This setup.py is for building a legacy toast-cmb package on PyPI.

import os
import sys

from setuptools import setup


conf = dict()
conf["name"] = "toast-cmb"
conf["description"] = "ALIAS to Time Ordered Astrophysics Scalable Tools"
conf["long_description"] = "This package is a deprecated alias.  Use https://pypi.org/project/toast instead."
conf["long_description_content_type"] = "text/markdown"
conf["author"] = "Theodore Kisner, Reijo Keskitalo"
conf["author_email"] = "tskisner.public@gmail.com"
conf["license"] = "BSD"
conf["url"] = "https://github.com/hpc4cmb/toast"
conf["version"] = "2.3.14"
conf["python_requires"] = ">=3.7.0"
conf["setup_requires"] = (["wheel"],)
conf["install_requires"] = ["toast",]
conf["platforms"] = "all"
conf["zip_safe"] = False
conf["classifiers"] = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering :: Astronomy",
]

setup(**conf)
