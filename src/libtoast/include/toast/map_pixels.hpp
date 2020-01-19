
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

    #pragma omp parallel for default(shared) schedule(static, 64)
    for (size_t i = 0; i < nsamp; ++i) {
        T pixel = global_pixels[i];
        T submap = 0;
        if (pixel < 0) {
            pixel = -1;
        } else {
            submap = static_cast <T> (
                static_cast <double> (pixel) * npix_submap_inv
                );
            pixel -= submap * static_cast <T> (npix_submap);
        }
        local_pixels[i] = pixel;
        submap = static_cast <T> (global2local[submap]);
        local_submaps[i] = submap;
    }
    return;
}
}

#endif // ifndef TOAST_MAP_PIXELS_HPP
