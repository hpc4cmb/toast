# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

import sys

from ..ctoast import test_runner


def test_ctoast(testdir):
    ret = test_runner(sys.argc, sys.argv)
    return

