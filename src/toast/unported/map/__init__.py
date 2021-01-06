# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

# import functions in our public API

from .pixels import DistPixels

from .cov import (
    covariance_invert,
    covariance_rcond,
    covariance_multiply,
    covariance_apply,
)
