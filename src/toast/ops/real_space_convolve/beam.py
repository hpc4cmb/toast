# Copyright (c) 2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np


class RealSpaceBeam(object):
    """A beam discretized as weighted nodes around the beam center.

    Each node is a rotation from the beam center to the node, in the detector
    frame.  Each node has a weight for I and a weight shared by Q and U.

    Args:
        quats (array):  The node quaternions, shape (n_node, 4).
        weights_I (array):  The I weight of each node, shape (n_node,).
        weights_P (array):  The Q and U weight of each node, shape (n_node,), or
            None for an intensity-only beam.

    """

    def __init__(self, quats, weights_I, weights_P=None):
        self.quats = np.ascontiguousarray(quats, dtype=np.float64)
        self.weights_I = np.ascontiguousarray(weights_I, dtype=np.float64)
        self.weights_P = None
        if weights_P is not None:
            self.weights_P = np.ascontiguousarray(weights_P, dtype=np.float64)
        n_node = len(self.quats)
        if self.quats.shape != (n_node, 4):
            msg = f"Beam quaternions have shape {self.quats.shape}, expected (n_node, 4)"
            raise ValueError(msg)
        for name in ("weights_I", "weights_P"):
            weights = getattr(self, name)
            if weights is not None and weights.shape != (n_node,):
                msg = f"Beam {name} have shape {weights.shape}, expected ({n_node},)"
                raise ValueError(msg)

    @classmethod
    def load(cls, path):
        """Read a beam from a `.npz` file written by `save`.

        Args:
            path (str):  The file to read.

        Returns:
            (RealSpaceBeam):  The beam.

        """
        with np.load(path) as data:
            weights_P = data["weights_P"] if "weights_P" in data.files else None
            return cls(data["quats"], data["weights_I"], weights_P)

    def save(self, path):
        """Write the beam to a `.npz` file.

        Args:
            path (str):  The file to write.

        Returns:
            None

        """
        arrays = {"quats": self.quats, "weights_I": self.weights_I}
        if self.weights_P is not None:
            arrays["weights_P"] = self.weights_P
        np.savez(path, **arrays)
