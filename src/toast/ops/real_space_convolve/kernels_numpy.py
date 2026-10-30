# Copyright (c) 2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import healpy as hp
import numpy as np

from ... import qarray as qa
from ...accelerator import ImplementationType, kernel

# Number of stencil entries (4 pixels per node and sample) in a chunk of samples.
_CHUNK_ENTRIES = 2**15

_XAXIS = np.array([1.0, 0.0, 0.0])
_YAXIS = np.array([0.0, 1.0, 0.0])
_ZAXIS = np.array([0.0, 0.0, 1.0])


def _spin2_transport(z_pix, sth_pix, phi_pix, z_los, sth_los, phi_los):
    """Spin-2 parallel transport from pixel centers to the line of sight.

    The transport angle is delta = alpha - gamma, with alpha the bearing from the
    pixel center towards the line of sight and gamma the bearing from the line of
    sight back towards the pixel center, both east of north.  It is computed from
    the haversine of the separation, which stays accurate at small separations.

    Args:
        z_pix (array):  Cosine of the colatitude of the pixel centers.
        sth_pix (array):  Sine of the colatitude of the pixel centers.
        phi_pix (array):  Longitude of the pixel centers.
        z_los (array):  Cosine of the colatitude of the line of sight.
        sth_los (array):  Sine of the colatitude of the line of sight.
        phi_los (array):  Longitude of the line of sight.

    Returns:
        (tuple):  (cos 2 delta, sin 2 delta).

    """
    dphi = phi_los - phi_pix
    ch = np.cos(0.5 * dphi)
    sh = np.sin(0.5 * dphi)
    sin_dphi = 2.0 * ch * sh
    sh2 = sh * sh
    cos_dphi = 1.0 - 2.0 * sh2

    dz = z_los - z_pix
    ds = sth_los - sth_pix
    st2 = 0.25 * (dz * dz + ds * ds)
    prod_sth = sth_pix * sth_los
    hav = st2 + prod_sth * sh2

    num = 2.0 * sin_dphi * (z_los + z_pix) * hav
    den = (
        4.0 * st2 * (1.0 - st2) * cos_dphi
        - 4.0 * prod_sth * z_los * z_pix * sh2 * sh2
        + prod_sth * sin_dphi * sin_dphi
    )

    norm = num * num + den * den
    same = norm == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / norm
        cos_2d = np.where(same, 1.0, (den * den - num * num) * inv)
        sin_2d = np.where(same, 0.0, -2.0 * num * den * inv)
    return cos_2d, sin_2d


def _stencil(nside, rot_x, rot_y, los, node_dirs):
    """Bilinear RING pixels and weights of every node at each sample.

    Args:
        nside (int):  The map NSIDE.
        rot_x (array):  The detector X axis at each sample, shape (n_samp, 3).
        rot_y (array):  The detector Y axis at each sample, shape (n_samp, 3).
        los (array):  The line of sight at each sample, shape (n_samp, 3).
        node_dirs (array):  The node directions in the detector frame, (n_node, 3).

    Returns:
        (tuple):  The pixels and weights, each of shape (4, n_samp, n_node).

    """
    # Rotate the node directions by each detector quaternion.
    vec = (
        rot_x[:, None, :] * node_dirs[None, :, 0, None]
        + rot_y[:, None, :] * node_dirs[None, :, 1, None]
        + los[:, None, :] * node_dirs[None, :, 2, None]
    )
    theta = np.arctan2(np.hypot(vec[..., 0], vec[..., 1]), vec[..., 2])
    phi = np.arctan2(vec[..., 1], vec[..., 0])
    pix, wts = hp.get_interp_weights(nside, theta.ravel(), phi.ravel())
    shape = (4, len(los), len(node_dirs))
    return pix.reshape(shape), wts.reshape(shape)


def _beam_sum(quats, nside, maps, node_dirs, weights_I, weights_P):
    """Convolve the map with the beam at each sample.

    Each node direction is read from the RING map with bilinear interpolation.  Q
    and U of each of the four pixels are transported to the line of sight before
    the weighted sum, so the result is in the meridian basis of the line of sight.

    Args:
        quats (array):  The detector quaternions, shape (n_samp, 4).
        nside (int):  The map NSIDE.
        maps (array):  The RING map, shape (1, n_pix) for I or (3, n_pix) for IQU.
        node_dirs (array):  The node directions in the detector frame, (n_node, 3).
        weights_I (array):  The I node weights.
        weights_P (array):  The Q and U node weights, unused for an I map.

    Returns:
        (array):  The beam-convolved Stokes parameters, shape (n_map, n_samp).

    """
    n_samp = len(quats)
    n_map = maps.shape[0]
    result = np.zeros((n_map, n_samp), dtype=np.float64)
    n_chunk = max(1, _CHUNK_ENTRIES // (4 * len(node_dirs)))

    for first in range(0, n_samp, n_chunk):
        last = min(first + n_chunk, n_samp)
        q = quats[first:last]
        rot_x = qa.rotate(q, _XAXIS)
        rot_y = qa.rotate(q, _YAXIS)
        los = qa.rotate(q, _ZAXIS)

        pix, wts = _stencil(nside, rot_x, rot_y, los, node_dirs)
        result[0, first:last] = np.einsum("jsn,jsn,n->s", wts, maps[0][pix], weights_I)
        if n_map == 1:
            continue

        theta_pix, phi_pix = hp.pix2ang(nside, pix)
        cos_2d, sin_2d = _spin2_transport(
            np.cos(theta_pix),
            np.sin(theta_pix),
            phi_pix,
            los[None, :, None, 2],
            np.hypot(los[:, 0], los[:, 1])[None, :, None],
            np.arctan2(los[:, 1], los[:, 0])[None, :, None],
        )

        # Q + iU of each pixel, multiplied by exp(2i delta).
        q_pix = maps[1][pix]
        u_pix = maps[2][pix]
        result[1, first:last] = np.einsum(
            "jsn,jsn,n->s", wts, q_pix * cos_2d - u_pix * sin_2d, weights_P
        )
        result[2, first:last] = np.einsum(
            "jsn,jsn,n->s", wts, q_pix * sin_2d + u_pix * cos_2d, weights_P
        )
    return result


@kernel(impl=ImplementationType.NUMPY, name="real_space_convolve")
def real_space_convolve_numpy(
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
    node_dirs = qa.rotate(node_quats, _ZAXIS)
    for idet in range(len(quat_index)):
        dquats = quats[quat_index[idet]]
        dweights = weights[weight_index[idet]].reshape((len(dquats), -1))
        tod = det_data[det_data_index[idet]]
        for vw in intervals:
            vslice = slice(vw.first, vw.last)
            stokes = _beam_sum(
                dquats[vslice],
                nside,
                maps,
                node_dirs,
                node_weights_I,
                node_weights_P,
            )
            tod[vslice] += np.sum(dweights[vslice].T * stokes, axis=0)
