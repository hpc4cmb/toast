# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Tools for creating fake HWP signals."""

import numpy as np

from ...observation import default_values as defaults


def fake_hwpss_data(ang, scale):
    # Generate a timestream of fake HWPSS
    n_harmonic = 5
    coscoeff = scale * np.array([1.0, 0.6, 0.2, 0.001, 0.003])
    sincoeff = scale * np.array([0.7, 0.9, 0.1, 0.002, 0.0005])
    out = np.zeros_like(ang)
    for h in range(n_harmonic):
        out[:] += coscoeff[h] * np.cos((h + 1) * ang) + sincoeff[h] * np.sin(
            (h + 1) * ang
        )
    return out, coscoeff, sincoeff


def fake_hwpss(data, field, scale):
    # Create a fake HWP synchronous signal
    coeff = dict()
    for ob in data.obs:
        hwpss, ccos, csin = fake_hwpss_data(ob.shared[defaults.hwp_angle].data, scale)
        n_harm = len(ccos)
        coeff[ob.name] = np.zeros(4 * n_harm)
        for h in range(n_harm):
            coeff[ob.name][4 * h] = csin[h]
            coeff[ob.name][4 * h + 1] = 0
            coeff[ob.name][4 * h + 2] = ccos[h]
            coeff[ob.name][4 * h + 3] = 0
        if field not in ob.detdata:
            ob.detdata.create(field, units=defaults.det_data_units)
        for det in ob.local_detectors:
            ob.detdata[field][det, :] += hwpss
    return coeff
