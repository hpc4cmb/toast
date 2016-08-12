#!/usr/bin/env python

import glob
import os
import sys
import re
import subprocess

import unittest

from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize

from setuptools.command.test import test as TestCommand

import numpy as np

from toast.mpirunner import MPITestRunner


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


# the core library of C functions

ctoast_dir = os.path.join(os.getcwd(), 'toast', 'ctoast')

libctoast = ('ctoast', 
    {
    'sources': [
        'toast/ctoast/pytoast_mem.c',
        'toast/ctoast/pytoast_qarray.c',
        'toast/ctoast/pytoast_rng.c',
        ],
    'include_dirs': [ctoast_dir, os.path.join(ctoast_dir, 'Random123')],
    }
)


# extensions to build

ext_map_noise = Extension (
    'toast.map._noise',
    include_dirs = [np.get_include(), ctoast_dir], 
    sources = [
        'toast/map/_noise.pyx'
    ]
)

ext_cache = Extension (
    'toast._cache',
    include_dirs = [np.get_include(), ctoast_dir],
    sources = [
        'toast/_cache.pyx'
    ]
)

ext_rng = Extension (
    'toast._rng',
    include_dirs = [np.get_include(), ctoast_dir],
    sources = [
        'toast/_rng.pyx'
    ]
)

ext_qarray = Extension (
    'toast.tod._qarray',
    include_dirs = [np.get_include(), ctoast_dir],
    sources = [
        'toast/tod/_qarray.pyx'
    ]
)

ext_psd_tools = Extension (
    'toast.fod._psd_tools',
    include_dirs = [np.get_include(), ctoast_dir],
    sources = [
        'toast/fod/_psd_tools.pyx'
    ],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

extensions = cythonize([
    ext_map_noise,
    ext_cache,
    ext_qarray,
    ext_psd_tools,
    ext_rng
])


# scripts to install

scripts = glob.glob('pipelines/*.py')

# customize the test command, to use MPI runner

class MPITestCommand(TestCommand):

    def __init__(self, *args, **kwargs):
        super(MPITestCommand, self).__init__(*args, **kwargs)

    def initialize_options(self):
        TestCommand.initialize_options(self)

    def finalize_options(self):
        TestCommand.finalize_options(self)
        #self.test_args = []
        self.test_suite = True

    def run(self):
        loader = unittest.TestLoader()
        runner = MPITestRunner(verbosity=2)
        suite = loader.discover('tests', pattern='test_*.py', top_level_dir='.')
        runner.run(suite)


# set it all up

setup (
    name = 'toast',
    provides = 'toast',
    version = current_version,
    description = 'Time Ordered Astrophysics Scalable Tools',
    author = 'Theodore Kisner',
    author_email = 'mail@theodorekisner.com',
    url = 'https://github.com/tskisner/pytoast',
    libraries = [libctoast],
    ext_modules = extensions,
    packages = ['toast', 'toast.tod', 'toast.map'],
    scripts = scripts,
    license = 'BSD',
    requires = ['Python (>3.4.0)', ],
    cmdclass={'test': MPITestCommand}
)


# extra cleanup of build products

if "clean" in sys.argv:
    # note that shell=True should be OK here because the command is constant.
    subprocess.call("rm -rf build", shell=True, executable="/bin/bash")
    subprocess.call("rm -rf dist", shell=True, executable="/bin/bash")
    subprocess.call("rm -rf toast/tod/*.so", shell=True, executable="/bin/bash")
    subprocess.call("rm -rf toast/map/*.so", shell=True, executable="/bin/bash")
    subprocess.call("rm -rf toast/*.so", shell=True, executable="/bin/bash")
    subprocess.call("rm -rf toast/tod/*.dylib", shell=True, executable="/bin/bash")
    subprocess.call("rm -rf toast/map/*.dylib", shell=True, executable="/bin/bash")
    subprocess.call("rm -rf toast/*.dylib", shell=True, executable="/bin/bash")
    subprocess.call("rm -rf test_output", shell=True, executable="/bin/bash")
    subprocess.call('find . -name "__pycache__" -exec rm -rf "{}" \; >/dev/null 2>&1', shell=True, executable="/bin/bash")
