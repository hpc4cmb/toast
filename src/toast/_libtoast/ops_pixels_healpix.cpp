// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <qarray.hpp>

#include <intervals.hpp>

#include <accelerator.hpp>

// 2/PI
#define TWOINVPI 0.63661977236758134308

// 2/3
#define TWOTHIRDS 0.66666666666666666667

// Helper table initialization
void hpix_init_utab(uint64_t *utab)
{
    for (uint64_t m = 0; m < 0x100; ++m)
    {
        utab[m] = (m & 0x1) | ((m & 0x2) << 1) | ((m & 0x4) << 2) |
                  ((m & 0x8) << 3) | ((m & 0x10) << 4) | ((m & 0x20) << 5) |
                  ((m & 0x40) << 6) | ((m & 0x80) << 7);
    }
    return;
}

#ifdef HAVE_OPENMP_TARGET
#pragma omp declare target
#endif // ifdef HAVE_OPENMP_TARGET

void pixels_healpix_qa_rotate(double const *q_in, double const *v_in,
                              double *v_out)
{
    // The input quaternion has already been normalized on the host.

    double xw = q_in[3] * q_in[0];
    double yw = q_in[3] * q_in[1];
    double zw = q_in[3] * q_in[2];
    double x2 = -q_in[0] * q_in[0];
    double xy = q_in[0] * q_in[1];
    double xz = q_in[0] * q_in[2];
    double y2 = -q_in[1] * q_in[1];
    double yz = q_in[1] * q_in[2];
    double z2 = -q_in[2] * q_in[2];

    v_out[0] = 2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] +
                    (yw + xz) * v_in[2]) +
               v_in[0];

    v_out[1] = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] +
                    (yz - xw) * v_in[2]) +
               v_in[1];

    v_out[2] = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] +
                    (x2 + y2) * v_in[2]) +
               v_in[2];

    return;
}

uint64_t hpix_xy2pix(uint64_t *utab, uint64_t x, uint64_t y)
{
    return utab[x & 0xff] | (utab[(x >> 8) & 0xff] << 16) |
           (utab[(x >> 16) & 0xff] << 32) |
           (utab[(x >> 24) & 0xff] << 48) |
           (utab[y & 0xff] << 1) | (utab[(y >> 8) & 0xff] << 17) |
           (utab[(y >> 16) & 0xff] << 33) |
           (utab[(y >> 24) & 0xff] << 49);
}

void hpix_vec2zphi(double const *vec, double *phi, int *region, double *z,
                   double *rtz)
{
    // region encodes BOTH the sign of Z and whether its
    // absolute value is greater than 2/3.
    (*z) = vec[2];
    double za = fabs(*z);
    int itemp = ((*z) > 0.0) ? 1 : -1;
    (*region) = (za <= TWOTHIRDS) ? itemp : itemp + itemp;
    (*rtz) = sqrt(3.0 * (1.0 - za));
    (*phi) = atan2(vec[1], vec[0]);
    return;
}

void hpix_zphi2nest(int64_t nside, int64_t factor, uint64_t *utab, double phi,
                    int region, double z, double rtz, int64_t *pix)
{
    double tt = (phi >= 0.0) ? phi * TWOINVPI : phi * TWOINVPI + 4.0;
    int64_t x;
    int64_t y;
    double temp1;
    double temp2;
    int64_t jp;
    int64_t jm;
    int64_t ifp;
    int64_t ifm;
    int64_t face;
    int64_t ntt;
    double tp;

    double dnside = static_cast<double>(nside);
    int64_t twonside = 2 * nside;
    double halfnside = 0.5 * dnside;
    double tqnside = 0.75 * dnside;
    int64_t nsideminusone = nside - 1;

    if ((region == 1) || (region == -1))
    {
        temp1 = halfnside + dnside * tt;
        temp2 = tqnside * z;

        jp = (int64_t)(temp1 - temp2);
        jm = (int64_t)(temp1 + temp2);

        ifp = jp >> factor;
        ifm = jm >> factor;

        if (ifp == ifm)
        {
            face = (ifp == 4) ? (int64_t)4 : ifp + 4;
        }
        else if (ifp < ifm)
        {
            face = ifp;
        }
        else
        {
            face = ifm + 8;
        }

        x = jm & nsideminusone;
        y = nsideminusone - (jp & nsideminusone);
    }
    else
    {
        ntt = (int64_t)tt;

        tp = tt - (double)ntt;

        temp1 = dnside * rtz;

        jp = (int64_t)(tp * temp1);
        jm = (int64_t)((1.0 - tp) * temp1);

        if (jp >= nside)
        {
            jp = nsideminusone;
        }
        if (jm >= nside)
        {
            jm = nsideminusone;
        }

        if (z >= 0)
        {
            face = ntt;
            x = nsideminusone - jm;
            y = nsideminusone - jp;
        }
        else
        {
            face = ntt + 8;
            x = jp;
            y = jm;
        }
    }

    uint64_t sipf = hpix_xy2pix(utab, (uint64_t)x, (uint64_t)y);

    (*pix) = (int64_t)sipf + (face << (2 * factor));

    return;
}

void hpix_zphi2ring(int64_t nside, int64_t factor, double phi, int region, double z,
                    double rtz, int64_t *pix)
{
    double tt = (phi >= 0.0) ? phi * TWOINVPI : phi * TWOINVPI + 4.0;
    double tp;
    int64_t longpart;
    double temp1;
    double temp2;
    int64_t jp;
    int64_t jm;
    int64_t ip;
    int64_t ir;
    int64_t kshift;

    double dnside = static_cast<double>(nside);
    int64_t fournside = 4 * nside;
    double halfnside = 0.5 * dnside;
    double tqnside = 0.75 * dnside;
    int64_t nsideplusone = nside + 1;
    int64_t ncap = 2 * (nside * nside - nside);
    int64_t npix = 12 * nside * nside;

    if ((region == 1) || (region == -1))
    {
        temp1 = halfnside + dnside * tt;
        temp2 = tqnside * z;

        jp = (int64_t)(temp1 - temp2);
        jm = (int64_t)(temp1 + temp2);

        ir = nsideplusone + jp - jm;
        kshift = 1 - (ir & 1);

        ip = (jp + jm - nside + kshift + 1) >> 1;
        ip = ip % fournside;

        (*pix) = ncap + ((ir - 1) * fournside + ip);
    }
    else
    {
        tp = tt - floor(tt);

        temp1 = dnside * rtz;

        jp = (int64_t)(tp * temp1);
        jm = (int64_t)((1.0 - tp) * temp1);
        ir = jp + jm + 1;
        ip = (int64_t)(tt * (double)ir);
        longpart = (int64_t)(ip / (4 * ir));
        ip -= longpart;

        (*pix) = (region > 0) ? (2 * ir * (ir - 1) + ip)
                              : (npix - 2 * ir * (ir + 1) + ip);
    }

    return;
}

void pixels_healpix_nest_inner(
    int64_t nside,
    int64_t factor,
    uint64_t *utab,
    int32_t const *quat_index,
    int32_t const *pixel_index,
    double const *quats,
    uint8_t const *flags,
    uint8_t *hsub,
    int64_t *pixels,
    int64_t n_pix_submap,
    int64_t isamp,
    int64_t n_samp,
    int64_t idet,
    uint8_t mask,
    bool use_flags)
{
    const double zaxis[3] = {0.0, 0.0, 1.0};
    int32_t p_indx = pixel_index[idet];
    int32_t q_indx = quat_index[idet];
    double dir[3];
    double z;
    double rtz;
    double phi;
    int region;
    size_t qoff = (q_indx * 4 * n_samp) + 4 * isamp;
    size_t poff = p_indx * n_samp + isamp;
    int64_t sub_map;

    pixels_healpix_qa_rotate(&(quats[qoff]), zaxis, dir);
    hpix_vec2zphi(dir, &phi, &region, &z, &rtz);
    hpix_zphi2nest(nside, factor, utab, phi, region, z, rtz, &(pixels[poff]));
    if (use_flags && ((flags[isamp] & mask) != 0))
    {
        pixels[poff] = -1;
    }
    else
    {
        sub_map = (int64_t)(pixels[poff] / n_pix_submap);
        hsub[sub_map] = 1;
    }

    return;
}

void pixels_healpix_ring_inner(
    int64_t nside,
    int64_t factor,
    int32_t const *quat_index,
    int32_t const *pixel_index,
    double const *quats,
    uint8_t const *flags,
    uint8_t *hsub,
    int64_t *pixels,
    int64_t n_pix_submap,
    int64_t isamp,
    int64_t n_samp,
    int64_t idet,
    uint8_t mask,
    bool use_flags)
{
    const double zaxis[3] = {0.0, 0.0, 1.0};
    int32_t p_indx = pixel_index[idet];
    int32_t q_indx = quat_index[idet];
    double dir[3];
    double z;
    double rtz;
    double phi;
    int region;
    size_t qoff = (q_indx * 4 * n_samp) + 4 * isamp;
    size_t poff = p_indx * n_samp + isamp;
    int64_t sub_map;

    pixels_healpix_qa_rotate(&(quats[qoff]), zaxis, dir);
    hpix_vec2zphi(dir, &phi, &region, &z, &rtz);
    hpix_zphi2ring(nside, factor, phi, region, z, rtz, &(pixels[poff]));
    if (use_flags && ((flags[isamp] & mask) != 0))
    {
        pixels[poff] = -1;
    }
    else
    {
        sub_map = (int64_t)(pixels[poff] / n_pix_submap);
        hsub[sub_map] = 1;
    }

    return;
}

#ifdef HAVE_OPENMP_TARGET
#pragma omp end declare target
#endif // ifdef HAVE_OPENMP_TARGET

void init_ops_pixels_healpix(py::module &m)
{
    m.def(
        "pixels_healpix", [](
                              py::buffer quat_index,
                              py::buffer quats,
                              py::buffer shared_flags,
                              uint8_t shared_flag_mask,
                              py::buffer pixel_index,
                              py::buffer pixels,
                              py::buffer intervals,
                              py::buffer hit_submaps,
                              int64_t n_pix_submap,
                              int64_t nside,
                              bool nest,
                              bool use_accel)
        {
            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = (!omgr.device_is_host()) && use_accel;

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(3);

            int32_t * raw_quat_index = extract_buffer <int32_t> (
                quat_index, "quat_index", 1, temp_shape, {-1}
            );
            int64_t n_det = temp_shape[0];

            int32_t * raw_pixel_index = extract_buffer <int32_t> (
                pixel_index, "pixel_index", 1, temp_shape, {n_det}
            );

            int64_t * raw_pixels = extract_buffer <int64_t> (
                pixels, "pixels", 2, temp_shape, {-1, -1}
            );
            int64_t n_samp = temp_shape[1];

            double * raw_quats = extract_buffer <double> (
                quats, "quats", 3, temp_shape, {-1, n_samp, 4}
            );

            Interval * raw_intervals = extract_buffer <Interval> (
                intervals, "intervals", 1, temp_shape, {-1}
            );
            int64_t n_view = temp_shape[0];

            uint8_t * raw_hsub = extract_buffer <uint8_t> (
                hit_submaps, "hit_submaps", 1, temp_shape, {-1}
            );
            int64_t n_submap = temp_shape[0];

            // Optionally use flags
            bool use_flags = true;
            uint8_t * raw_flags = extract_buffer <uint8_t> (
                shared_flags, "flags", 1, temp_shape, {-1}
            );
            if (temp_shape[0] != n_samp) {
                raw_flags = omgr.null_ptr <uint8_t> ();
                use_flags = false;
            }

            static uint64_t utab[0x100];
            hpix_init_utab(utab);

            int64_t factor = 0;
            while (nside != (1ll << factor)) {
                ++factor;
            }

            if (offload) {
#ifdef HAVE_OPENMP_TARGET

                double * dev_quats = omgr.device_ptr(raw_quats);
                int64_t * dev_pixels = omgr.device_ptr(raw_pixels);
                Interval * dev_intervals = omgr.device_ptr(raw_intervals);
                uint8_t * dev_flags = omgr.device_ptr(raw_flags);

                // Make sure the lookup table exists on device
                size_t utab_bytes = 0x100 * sizeof(uint64_t);
                int present_utab = omgr.present(utab, utab_bytes);
                if (present_utab == 0) {
                    void * vptr = omgr.create((void *)utab, utab_bytes);
                    omgr.update_device((void *)utab, utab_bytes);
                }

                uint64_t * dev_utab = omgr.device_ptr(utab);

 // Calculate the maximum interval size on the CPU
int64_t max_interval_size = 0;
for (int64_t iview = 0; iview < n_view; iview++) {
    int64_t interval_size = raw_intervals[iview].last - raw_intervals[iview].first + 1;
    if (interval_size > max_interval_size) {
        max_interval_size = interval_size;
    }
}

#pragma omp target data map(to : raw_pixel_index[0 : n_det], \
                                raw_quat_index[0 : n_det],   \
                                n_pix_submap,                \
                                nside,                       \
                                factor,                      \
                                nest,                        \
                                n_view,                      \
                                n_det,                       \
                                n_samp,                      \
                                shared_flag_mask,            \
                                use_flags)                   \
    map(tofrom : raw_hsub[0 : n_submap])
{
    if (nest) {
#pragma omp target teams distribute parallel for collapse(3)
        for (int64_t idet = 0; idet < n_det; idet++) {
            for (int64_t iview = 0; iview < n_view; iview++) {
                for (int64_t isamp = 0; isamp < max_interval_size; isamp++) {
                    // Adjust for the actual start of the interval
                    int64_t adjusted_isamp = isamp + dev_intervals[iview].first;

                    // Check if the value is out of range for the current interval
                    if (adjusted_isamp > dev_intervals[iview].last) {
                        continue;
                    }

                    pixels_healpix_nest_inner(
                        nside,
                        factor,
                        dev_utab,
                        raw_quat_index,
                        raw_pixel_index,
                        dev_quats,
                        dev_flags,
                        raw_hsub,
                        dev_pixels,
                        n_pix_submap,
                        adjusted_isamp,
                        n_samp,
                        idet,
                        shared_flag_mask,
                        use_flags
                    );
                }
            }
        }
    } else {
#pragma omp target teams distribute parallel for collapse(3)
        for (int64_t idet = 0; idet < n_det; idet++) {
            for (int64_t iview = 0; iview < n_view; iview++) {
                for (int64_t isamp = 0; isamp < max_interval_size; isamp++) {
                    // Adjust for the actual start of the interval
                    int64_t adjusted_isamp = isamp + dev_intervals[iview].first;

                    // Check if the value is out of range for the current interval
                    if (adjusted_isamp > dev_intervals[iview].last) {
                        continue;
                    }

                    pixels_healpix_ring_inner(
                        nside,
                        factor,
                        raw_quat_index,
                        raw_pixel_index,
                        dev_quats,
                        dev_flags,
                        raw_hsub,
                        dev_pixels,
                        n_pix_submap,
                        adjusted_isamp,
                        n_samp,
                        idet,
                        shared_flag_mask,
                        use_flags
                    );
                }
            }
        }
    }
}

#endif // ifdef HAVE_OPENMP_TARGET
            } else {
                if (nest) {
                    for (int64_t idet = 0; idet < n_det; idet++) {
                        for (int64_t iview = 0; iview < n_view; iview++) {
#pragma omp parallel for default(shared)
                            for (
                                int64_t isamp = raw_intervals[iview].first;
                                isamp <= raw_intervals[iview].last;
                                isamp++
                            ) {
                                pixels_healpix_nest_inner(
                                    nside,
                                    factor,
                                    utab,
                                    raw_quat_index,
                                    raw_pixel_index,
                                    raw_quats,
                                    raw_flags,
                                    raw_hsub,
                                    raw_pixels,
                                    n_pix_submap,
                                    isamp,
                                    n_samp,
                                    idet,
                                    shared_flag_mask,
                                    use_flags
                                );
                            }
                        }
                    }
                } else {
                    for (int64_t idet = 0; idet < n_det; idet++) {
                        for (int64_t iview = 0; iview < n_view; iview++) {
#pragma omp parallel for default(shared)
                            for (
                                int64_t isamp = raw_intervals[iview].first;
                                isamp <= raw_intervals[iview].last;
                                isamp++
                            ) {
                                pixels_healpix_ring_inner(
                                    nside,
                                    factor,
                                    raw_quat_index,
                                    raw_pixel_index,
                                    raw_quats,
                                    raw_flags,
                                    raw_hsub,
                                    raw_pixels,
                                    n_pix_submap,
                                    isamp,
                                    n_samp,
                                    idet,
                                    shared_flag_mask,
                                    use_flags
                                );
                            }
                        }
                    }
                }
            }
            return; });
}
