# Copyright (c) 2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import healpy as hp
import numpy as np

from ... import qarray as qa
from ..._libtoast import ops_real_space_convolve as libtoast_real_space_convolve
from ...accelerator import ImplementationType, kernel
from .kernels_numpy import real_space_convolve_numpy


@kernel(impl=ImplementationType.DEFAULT)
def real_space_convolve(
    quat_index,
    quats,
    weight_index,
    weights,
    det_data_index,
    det_data,
    intervals,
    nside,
    maps,
    node_quats,
    node_weights_I,
    node_weights_P,
    use_accel=False,
):
    """Kernel for convolving a map with a discretized beam into timestreams.

    At each sample, every beam node is read from the map with bilinear
    interpolation, Q and U are transported to the line of sight, and the weighted
    sum over the nodes is contracted with the Stokes weights and added to the
    detector data.

    Args:
        quat_index (array):  The index into the detector quaternion array for each
            detector.
        quats (array):  The array of detector quaternions for each sample.
        weight_index (array):  The index into the weights array for each detector.
        weights (array):  The array of Stokes weights at each sample for each
            detector, with as many weights as map components.
        det_data_index (array):  The index into the data array for each detector.
        det_data (array):  The detector data at each sample for each detector.
        intervals (array):  The array of sample intervals.
        nside (int):  The NSIDE of the map.
        maps (array):  The full-sky RING map, shape (1, n_pix) for I or (3, n_pix)
            for IQU.
        node_quats (array):  The beam node quaternions, shape (n_node, 4).
        node_weights_I (array):  The I weight of each beam node.
        node_weights_P (array):  The Q and U weight of each beam node, unused for
            an I map.
        use_accel (bool):  Whether to use the accelerator for this call (if supported).

    Returns:
        None.

    """
    return _real_space_convolve_compiled(
        quat_index,
        quats,
        weight_index,
        weights,
        det_data_index,
        det_data,
        intervals,
        nside,
        maps,
        node_quats,
        node_weights_I,
        node_weights_P,
        use_accel=use_accel,
    )


@kernel(impl=ImplementationType.COMPILED, name="real_space_convolve")
def real_space_convolve_compiled(*args, use_accel=False):
    return _real_space_convolve_compiled(*args, use_accel=use_accel)


def _real_space_convolve_compiled(
    quat_index,
    quats,
    weight_index,
    weights,
    det_data_index,
    det_data,
    intervals,
    nside,
    maps,
    node_quats,
    node_weights_I,
    node_weights_P,
    use_accel=False,
):
    if use_accel:
        raise NotImplementedError("real_space_convolve has no accelerator support")
    if weights.ndim == 2:
        # Intensity-only weights have one value per sample, but the compiled
        # kernel expects a 3D array.
        weights = weights.reshape(weights.shape + (1,))
    node_dirs = qa.rotate(node_quats, np.array([0.0, 0.0, 1.0]))
    # Consecutive nodes in NESTED pixel order are close on the sky, so they share
    # map pixels and the kernel reuses their transport.
    order = np.argsort(hp.vec2pix(2**20, *node_dirs.T, nest=True), kind="stable")
    if len(node_weights_P) > 0:
        node_weights_P = node_weights_P[order]
    libtoast_real_space_convolve(
        quat_index,
        quats,
        weight_index,
        weights,
        det_data_index,
        det_data,
        intervals,
        nside,
        maps,
        node_dirs[order],
        node_weights_I[order],
        node_weights_P,
    )
