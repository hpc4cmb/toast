
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_TOD_SIMMAP_HPP
#define TOAST_TOD_SIMMAP_HPP

namespace toast {

template <typename T>
void scan_map(int64_t const * submap, int64_t subnpix, double const * weights,
              int64_t nmap, int64_t * subpix, T const * map, double * tod,
              int64_t nsamp) {
    #pragma \
    omp parallel for schedule(static) default(none) shared(submap, subnpix, weights, nmap, subpix, map, tod, nsamp)
    for (int64_t i = 0; i < nsamp; ++i) {
        tod[i] = 0.0;
        int64_t offset = (submap[i] * subnpix + subpix[i]) * nmap;
        int64_t woffset = i * nmap;
        for (int64_t imap = 0; imap < nmap; ++imap) {
            tod[i] += map[offset++] * weights[woffset++];
        }
    }

    return;
}

}

#endif // ifndef TOAST_TOD_SIMMAP_HPP
