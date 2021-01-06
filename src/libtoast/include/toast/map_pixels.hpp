
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MAP_PIXELS_HPP
#define TOAST_MAP_PIXELS_HPP


namespace toast {
template <typename T>
void global_to_local(size_t nsamp,
                     T const * global_pixels,
                     size_t npix_submap,
                     int64_t const * global2local,
                     T * local_submaps,
                     T * local_pixels) {
    double npix_submap_inv = 1.0 / static_cast <double> (npix_submap);

    // Note:  there is not much work in this loop, so it might benefit from
    // omp simd instead.  However, that would only be enabled if the input
    // memory buffers were aligned.  That could be ensured with care in the
    // calling code.  To be revisited if this code is ever the bottleneck.

    #pragma omp parallel for default(shared) schedule(static)
    for (size_t i = 0; i < nsamp; ++i) {
        if (global_pixels[i] < 0) {
            local_submaps[i] = -1;
            local_pixels[i] = -1;
        } else {
            local_pixels[i] = global_pixels[i] % npix_submap;
            local_submaps[i] = static_cast <T> (
                global2local[
                    static_cast <T> (
                        static_cast <double> (global_pixels[i]) * npix_submap_inv
                        )
                ]
                );
        }
    }

    return;
}
}

#endif // ifndef TOAST_MAP_PIXELS_HPP
