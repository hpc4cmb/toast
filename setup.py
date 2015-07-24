#!/usr/bin/env python

import glob
import os
import sys
import re
import subprocess

from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize


def get_version():
    ver = 'unknown'
    if os.path.isfile("toast/_version.py"):
        f = open("toast/_version.py", "r")
        for line in f.readlines():
            mo = re.match("__version__ = '(.*)'", line)
            if mo:
                ver = mo.group(1)
        f.close()
    return ver

current_version = get_version()


# extensions to build

extensions = cythonize ( [] )


# scripts to install

scripts = glob.glob('pipelines/*.py')

# set it all up

setup (
    name = 'toast',
    provides = 'toast',
    version = current_version,
    description = 'Time Ordered Astrophysics Scalable Tools',
    author = 'Theodore Kisner',
    author_email = 'mail@theodorekisner.com',
    url = 'https://github.com/tskisner/pytoast',
    ext_modules = extensions,
    packages = [ 'toast' ],
    scripts = scripts,
    license = 'BSD',
    requires = ['Python (>3.4.0)', ],
    test_suite = 'tests.suite'
)


# extra cleanup of cython generated sources

if "clean" in sys.argv:
    print("Deleting cython files...")
    # Just in case the build directory was created by accident,
    # note that shell=True should be OK here because the command is constant.
    subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf pympit/*.c", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf pympit/*.so", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf pympit/*.pyc", shell=True, executable="/bin/bash")

