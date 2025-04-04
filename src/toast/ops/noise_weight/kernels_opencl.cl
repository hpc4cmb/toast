// Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

__kernel void noise_weight(
    int n_det,
    long n_sample,
    long first_sample,
    __global double const * weights,
    __global int const * det_data_index,
    __global double * det_data
) {
    // Get the global index of this work element
    int idet = get_global_id(0);
    long isamp = first_sample + get_global_id(1);

    int didx = det_data_index[idet];

    det_data[didx * n_sample + isamp] *= weights[idet];
}

