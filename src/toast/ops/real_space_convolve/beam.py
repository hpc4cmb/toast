# Copyright (c) 2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

_VALID_STOKES = ("I", "QU", "IQU")


class RealSpaceBeam(object):
    """A beam discretized as weighted nodes around the beam center.

    Each node is a rotation from the beam center to the node, in the detector
    frame.  The weights are shared by every Stokes component, or given per
    component, in which case the Q and U weights must be equal.

    Args:
        quats (array):  The node quaternions, shape (n_node, 4).
        weights (array):  The node weights, shape (n_node,), or (n_comp, n_node)
            with one row per component of `stokes`.
        stokes (str):  Empty for shared weights, otherwise "I", "QU" or "IQU".

    """

    def __init__(self, quats, weights, stokes=""):
        quats = np.ascontiguousarray(quats, dtype=np.float64)
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        stokes = str(stokes)
        if quats.ndim != 2 or quats.shape[1] != 4 or quats.shape[0] == 0:
            raise ValueError(
                f"Beam quaternions have shape {quats.shape}, expected (n_node, 4)"
            )
        n_node = quats.shape[0]
        if not np.all(np.isfinite(quats)) or not np.all(np.isfinite(weights)):
            raise ValueError("Beam quaternions and weights must be finite")
        norm = np.linalg.norm(quats, axis=1)
        if np.max(np.abs(norm - 1.0)) > 1.0e-12:
            raise ValueError("Beam quaternions must have unit norm")
        if stokes == "":
            if weights.shape != (n_node,):
                raise ValueError(
                    f"Shared beam weights have shape {weights.shape}, expected "
                    f"({n_node},)"
                )
        else:
            if stokes not in _VALID_STOKES:
                raise ValueError(
                    f"Beam Stokes components are '{stokes}', expected one of "
                    f"{_VALID_STOKES}"
                )
            if weights.shape != (len(stokes), n_node):
                raise ValueError(
                    f"Beam weights for '{stokes}' have shape {weights.shape}, "
                    f"expected ({len(stokes)}, {n_node})"
                )
            if "Q" in stokes:
                q_row = weights[stokes.index("Q")]
                u_row = weights[stokes.index("U")]
                if not np.array_equal(q_row, u_row):
                    raise ValueError(
                        "Beam weights of Q and U differ: unequal polarization beams "
                        "are not supported"
                    )
        self.quats = quats
        self.weights = weights
        self.stokes = stokes

    @property
    def n_node(self):
        """The number of beam nodes."""
        return self.quats.shape[0]

    @property
    def intensity_weights(self):
        """The node weights of I, or None if the beam has no I weights."""
        if self.stokes == "":
            return self.weights
        if "I" in self.stokes:
            return self.weights[self.stokes.index("I")]
        return None

    @property
    def polarization_weights(self):
        """The node weights of Q and U, or None if the beam has no Q and U weights."""
        if self.stokes == "":
            return self.weights
        if "Q" in self.stokes:
            return self.weights[self.stokes.index("Q")]
        return None

    @classmethod
    def load(cls, path):
        """Read a beam from a `.npz` file.

        The file holds `quats`, `weights` and, for per-component weights,
        `stokes`, as in the constructor.  Other arrays are ignored.

        Args:
            path (str):  The file to read.

        Returns:
            (RealSpaceBeam):  The beam.

        """
        with np.load(path) as data:
            missing = [x for x in ("quats", "weights") if x not in data.files]
            if len(missing) > 0:
                raise ValueError(
                    f"Beam file {path} is missing the arrays: {', '.join(missing)}"
                )
            quats = data["quats"]
            weights = data["weights"]
            stokes = str(data["stokes"]) if "stokes" in data.files else ""
        return cls(quats, weights, stokes=stokes)

    def save(self, path):
        """Write the beam to a `.npz` file readable by `load`.

        Args:
            path (str):  The file to write.

        Returns:
            None

        """
        np.savez(path, quats=self.quats, weights=self.weights, stokes=self.stokes)
