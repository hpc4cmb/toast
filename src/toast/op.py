# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


class Operator(object):
    """Base class for an operator that acts on collections of observations.

    An operator takes as input a toast.dist.Data object and returns a
    new instance of the same size.  For each observation in the distributed
    data, an operator may pass some data types forward unchanged, or it may
    replace or modify data.

    Args:
        None

    """

    def __init__(self):
        pass

    def exec(self, data):
        return data
