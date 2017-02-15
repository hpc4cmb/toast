# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

import os
import tempfile

from ..version import __version__

from .ctoast import test_ctoast



def test(testdir=None):
    # If no test output directory is specified, then create a temporary
    # one and delete it afterwards.
    outdir = None
    if testdir is None:
        outdir = tempfile.mkdtemp()
    else:
        outdir = os.path.abspath(testdir)
        os.makedirs(outdir)


    if testdir is None:
        os.rmdirs(outdir)


