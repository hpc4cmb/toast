# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import warnings

_matplotlib_backend = None


def set_matplotlib_backend(backend="pdf"):
    """Set the matplotlib backend."""
    global _matplotlib_backend
    if _matplotlib_backend is not None:
        return
    try:
        _matplotlib_backend = backend
        import matplotlib

        matplotlib.use(_matplotlib_backend, force=False)
    except:
        msg = "Could not set the matplotlib backend to '{}'".format(_matplotlib_backend)
        warnings.warn(msg)
