# Copyright (c) 2019-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os

from ..timing import function_timer, Timer
from ..utils import Logger, Environment


def add_debug_args(parser):
    # `debug` may be already added
    try:
        parser.add_argument(
            "--debug",
            required=False,
            action="store_true",
            help="Enable extra debugging outputs",
            dest="debug",
        )
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument(
            "--no-debug",
            required=False,
            action="store_false",
            help="Disable extra debugging outputs",
            dest="debug",
        )
    except argparse.ArgumentError:
        pass
    parser.set_defaults(debug=False)
    return
