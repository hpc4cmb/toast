# Copyright (c) 2015-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os


def filename_is_fits(filename):
    return filename.endswith((".fits", ".fit", ".FITS"))


def filename_is_hdf5(filename):
    return filename.endswith((".hdf", ".hdf5", ".h5", ".H5"))
