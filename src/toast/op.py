# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


class Operator(object):
    """Base class for an operator that acts on collections of observations.

    An operator takes as input a toast.dist.Data object and modifies it in place.

    Args:
        None

    """

    def __init__(self):
        pass

    def exec(self, data):
        """Perform operations on a Data object.

        Args:
            data (toast.Data):  The distributed data.

        Returns:
            None

        """
        return
