# Copyright (c) 2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

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
    node_quats_I,
    node_weights_I,
    node_quats_P,
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
        node_quats_I (array):  The quaternions of the I beam nodes, shape
            (n_node, 4).
        node_weights_I (array):  The weights of the I beam nodes.
        node_quats_P (array):  The quaternions of the Q and U beam nodes, unused
            for an I map.
        node_weights_P (array):  The weights of the Q and U beam nodes, unused for
            an I map.
        use_accel (bool):  Whether to use the accelerator for this call (if supported).

    Returns:
        None.

    """
    return real_space_convolve_numpy(
        quat_index,
        quats,
        weight_index,
        weights,
        det_data_index,
        det_data,
        intervals,
        nside,
        maps,
        node_quats_I,
        node_weights_I,
        node_quats_P,
        node_weights_P,
        impl=ImplementationType.NUMPY,
        use_accel=use_accel,
    )
