
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_TOD_MAPSCAN_HPP
#define TOAST_TOD_MAPSCAN_HPP

#include <cstring>

namespace toast {
template <typename T>
void scan_local_map(int64_t const * submap, int64_t subnpix, double const * weights,
                    int64_t nmap, int64_t * subpix, T const * map, double * tod,
                    int64_t nsamp) {
    // There is a single pixel vector valid for all maps.
    // The number of maps packed into "map" is the same as the number of weights
    // packed into "weights".
    //
    // The TOD is *NOT* set to zero, to allow accumulation.
    #pragma \
    omp parallel for schedule(static) default(none) shared(submap, subnpix, weights, nmap, subpix, map, tod, nsamp)
    for (int64_t i = 0; i < nsamp; ++i) {
        if ((subpix[i] < 0) || (submap[i] < 0)) {
            continue;
        }
        int64_t offset = (submap[i] * subnpix + subpix[i]) * nmap;
        int64_t woffset = i * nmap;
        for (int64_t imap = 0; imap < nmap; ++imap) {
            tod[i] += map[offset++] * weights[woffset++];
        }
    }

    return;
}

template <typename T>
void fast_scanning(double * toi, int64_t nsamp,
                   int64_t const * pixels, double const * weights,
                   int64_t nweight, T const * bmap) {
    memset(toi, 0, nsamp * sizeof(double));
    #pragma omp parallel for
    for (int64_t row = 0; row < nsamp; ++row) {
        int64_t offset = row * nweight;
        for (int64_t col = 0; col < nweight; ++col) {
            int64_t pix = pixels[offset];
            if (pix < 0) continue;
            double weight = weights[offset];
            toi[row] += bmap[pix] * weight;
            ++offset;
        }
    }
    return;
}

//
// template <typename T>
// void scan_global_map(int64_t npixmap, int64_t * pixels,
//                      int64_t nweightmap, double const * weights,
//                      int64_t nmap, T const * map, double * tod, int64_t nsamp) {
//     // The TOD is *NOT* set to zero, to allow accumulation.
//     // This function supports both scanning multiple weights for a single pixel and
//     // also scanning general pixel / weight pairs from each map for each sample.
//     if (nweightmap == nmap) {
//         // The number of maps packed into "map" are is the same as the number of
//         // weights packed into "weights".  We multiply every weight with its
//         // corresponding map value.
//         if (npixmap == nweightmap) {
//             // We have separate pixels for every weight
//
//
//         } else {
//             // We have one pixel for all maps.
//
//
//         }
//         #pragma \
//         omp parallel for schedule(static) default(none) shared(pixels, weights, nmap,
// map, tod, nsamp)
//         for (int64_t i = 0; i < nsamp; ++i) {
//             // Initialize to zero, since we are scanning all maps
//             tod[i] = 0.0;
//             if (pixels[i] < 0) {
//                 continue;
//             }
//             int64_t offset = pixels[i] * nmap;
//             int64_t woffset = i * nmap;
//             for (int64_t imap = 0; imap < nmap; ++imap) {
//                 tod[i] += map[offset++] * weights[woffset++];
//             }
//         }
//     } else if (nmap == 1) {
//         // We have multiple weights, but are only accumulating a single map.
//         // Use strided access.
//         #pragma \
//         omp parallel for schedule(static) default(none) shared(pixels, weights,
// nweight, map, tod, nsamp)
//         for (int64_t i = 0; i < nsamp; ++i) {
//             // Do NOT Initialize to zero, since we may accumulate multiple maps.
//             if (pixels[i] < 0) {
//                 continue;
//             }
//             int64_t offset = pixels[i] * nmap;
//             int64_t woffset = i * nmap;
//             for (int64_t imap = 0; imap < nmap; ++imap) {
//                 tod[i] += map[offset++] * weights[woffset++];
//             }
//         }
//     } else {
//         // This is not supported
//         auto here = TOAST_HERE();
//         auto log = toast::Logger::get();
//         std::string msg(
//             "global map scanning should have a single map or an equal number of maps
// and weights.");
//         log.error(msg.c_str(), here);
//         throw std::runtime_error(msg.c_str());
//     }
//
//
//     return;
// }
//
}

#endif // ifndef TOAST_TOD_MAPSCAN_HPP
