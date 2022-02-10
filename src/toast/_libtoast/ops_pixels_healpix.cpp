// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <qarray.hpp>

#include <accelerator.hpp>


// 2/PI
#define TWOINVPI 0.63661977236758134308

// 2/3
#define TWOTHIRDS 0.66666666666666666667

typedef struct {
    int64_t nside;
    int64_t npix;
    int64_t ncap;
    double dnside;
    int64_t twonside;
    int64_t fournside;
    int64_t nsideplusone;
    int64_t nsideminusone;
    double halfnside;
    double tqnside;
    int64_t factor;
    int64_t jr[12];
    int64_t jp[12];
    uint64_t utab[0x100];
    uint64_t ctab[0x100];
} hpix;

void hpix_init(hpix * hp, int64_t nside) {
    hp->nside = nside;
    hp->ncap = 2 * (nside * nside - nside);
    hp->npix = 12 * nside * nside;
    hp->dnside = static_cast <double> (nside);
    hp->twonside = 2 * nside;
    hp->fournside = 4 * nside;
    hp->nsideplusone = nside + 1;
    hp->halfnside = 0.5 * (hp->dnside);
    hp->tqnside = 0.75 * (hp->dnside);
    hp->factor = 0;
    hp->nsideminusone = nside - 1;
    while (nside != (1ll << hp->factor)) {
        ++hp->factor;
    }

    static const int64_t init_jr[12] = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    memcpy(hp->jr, init_jr, sizeof(init_jr));

    static const int64_t init_jp[12] = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};
    memcpy(hp->jp, init_jp, sizeof(init_jp));

    for (uint64_t m = 0; m < 0x100; ++m) {
        hp->utab[m] = (m & 0x1) | ((m & 0x2) << 1) | ((m & 0x4) << 2) |
                      ((m & 0x8) << 3) | ((m & 0x10) << 4) | ((m & 0x20) << 5) |
                      ((m & 0x40) << 6) | ((m & 0x80) << 7);

        hp->ctab[m] = (m & 0x1) | ((m & 0x2) << 7) | ((m & 0x4) >> 1) |
                      ((m & 0x8) << 6) | ((m & 0x10) >> 2) | ((m & 0x20) << 5) |
                      ((m & 0x40) >> 3) | ((m & 0x80) << 4);
    }
    return;
}

#pragma acc routine seq
uint64_t hpix_xy2pix(hpix * hp, uint64_t x, uint64_t y) {
    return hp->utab[x & 0xff] | (hp->utab[(x >> 8) & 0xff] << 16) |
           (hp->utab[(x >> 16) & 0xff] << 32) |
           (hp->utab[(x >> 24) & 0xff] << 48) |
           (hp->utab[y & 0xff] << 1) | (hp->utab[(y >> 8) & 0xff] << 17) |
           (hp->utab[(y >> 16) & 0xff] << 33) |
           (hp->utab[(y >> 24) & 0xff] << 49);
}

#pragma acc routine seq
void hpix_vec2zphi(hpix * hp, double const * vec,
                   double * phi, int * region, double * z,
                   double * rtz) {
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

#pragma acc routine seq
void hpix_zphi2nest(hpix * hp, double phi, int region, double z,
                    double rtz, int64_t * pix) {
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

    if ((region == 1) || (region == -1)) {
        temp1 = hp->halfnside + hp->dnside * tt;
        temp2 = hp->tqnside * z;

        jp = (int64_t)(temp1 - temp2);
        jm = (int64_t)(temp1 + temp2);

        ifp = jp >> hp->factor;
        ifm = jm >> hp->factor;

        if (ifp == ifm) {
            face = (ifp == 4) ? (int64_t)4 : ifp + 4;
        } else if (ifp < ifm) {
            face = ifp;
        } else {
            face = ifm + 8;
        }

        x = jm & hp->nsideminusone;
        y = hp->nsideminusone - (jp & hp->nsideminusone);
    } else {
        ntt = (int64_t)tt;

        tp = tt - (double)ntt;

        temp1 = hp->dnside * rtz;

        jp = (int64_t)(tp * temp1);
        jm = (int64_t)((1.0 - tp) * temp1);

        if (jp >= hp->nside) {
            jp = hp->nsideminusone;
        }
        if (jm >= hp->nside) {
            jm = hp->nsideminusone;
        }

        if (z >= 0) {
            face = ntt;
            x = hp->nsideminusone - jm;
            y = hp->nsideminusone - jp;
        } else {
            face = ntt + 8;
            x = jp;
            y = jm;
        }
    }

    uint64_t sipf = hpix_xy2pix(hp, (uint64_t)x, (uint64_t)y);

    (*pix) = (int64_t)sipf + (face << (2 * hp->factor));

    return;
}

#pragma acc routine seq
void hpix_zphi2ring(hpix * hp, double phi, int region, double z,
                    double rtz, int64_t * pix) {
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

    if ((region == 1) || (region == -1)) {
        temp1 = hp->halfnside + hp->dnside * tt;
        temp2 = hp->tqnside * z;

        jp = (int64_t)(temp1 - temp2);
        jm = (int64_t)(temp1 + temp2);

        ir = hp->nsideplusone + jp - jm;
        kshift = 1 - (ir & 1);

        ip = (jp + jm - hp->nside + kshift + 1) >> 1;
        ip = ip % hp->fournside;

        (*pix) = hp->ncap + ((ir - 1) * hp->fournside + ip);
    } else {
        tp = tt - floor(tt);

        temp1 = hp->dnside * rtz;

        jp = (int64_t)(tp * temp1);
        jm = (int64_t)((1.0 - tp) * temp1);
        ir = jp + jm + 1;
        ip = (int64_t)(tt * (double)ir);
        longpart = (int64_t)(ip / (4 * ir));
        ip -= longpart;

        (*pix) = (region > 0) ? (2 * ir * (ir - 1) + ip)
                 : (hp->npix - 2 * ir * (ir + 1) + ip);
    }

    return;
}

void init_ops_pixels_healpix(py::module & m) {
    m.def(
        "pixels_healpix", [](
            py::buffer quat_indx,
            py::buffer quats,
            py::buffer pixel_indx,
            py::buffer pixels,
            py::buffer shared_flags,
            py::buffer hit_submaps,
            int64_t n_pix_submap,
            uint8_t shared_flag_mask, int64_t nside, bool nest
        ) {
            auto info_pixindx = pixel_indx.request();
            int32_t * raw_pixindx = reinterpret_cast <int32_t *> (info_pixindx.ptr);

            auto info_pixels = pixels.request();
            int64_t * raw_pixels = reinterpret_cast <int64_t *> (info_pixels.ptr);

            auto info_quatindx = quat_indx.request();
            int32_t * raw_quatindx = reinterpret_cast <int32_t *> (info_quatindx.ptr);

            auto info_quats = quats.request();
            double * raw_quats = reinterpret_cast <double *> (info_quats.ptr);

            auto info_flags = shared_flags.request();
            uint8_t * raw_flags = reinterpret_cast <uint8_t *> (info_flags.ptr);

            auto info_hsub = hit_submaps.request();
            uint8_t * raw_hsub = reinterpret_cast <uint8_t *> (info_hsub.ptr);

            size_t n_det = info_pixindx.shape[0];
            if (info_quatindx.shape[0] != n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "pixel and quat indices do not have same number of dets";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_pixels.shape[0] < n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "det pixels have fewer detectors than the index";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            size_t n_samp = info_pixels.shape[1];
            if (info_flags.shape[0] != n_samp) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "shared flags do not have same number of samples as pixels";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_quats.shape[0] < n_det) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "det quats have fewer detectors than the index";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_quats.shape[1] != n_samp) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "det quats do not have same number of samples as pixels";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            if (info_quats.shape[2] != 4) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "det quats do not have 4 elements per sample";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            size_t len_pixels = info_pixels.shape[0] * n_samp;
            size_t len_flags = n_samp;
            size_t len_quats = info_quats.shape[0] * n_samp * 4;
            size_t n_submap = info_hsub.shape[0];

            hpix hp;
            hpix_init(&hp, nside);

            // if (nest) {
            //     #pragma \
            //     acc data copy(raw_hsub[:n_submap]) copyin(n_det, n_samp, shared_flag_mask, hp, n_pix_submap, raw_pixindx[:n_det], raw_quatindx[:n_det]) present(raw_pixels[:len_pixels], raw_flags[:len_flags], raw_quats[:len_quats])
            //     {
            //         if (fake_openacc()) {
            //             // Set all "present" data to point at the fake device pointers
            //             auto & fake = FakeMemPool::get();
            //             raw_pixels = (int64_t *)fake.device_ptr(raw_pixels);
            //             raw_flags = (uint8_t *)fake.device_ptr(raw_flags);
            //             raw_quats = (double *)fake.device_ptr(raw_quats);
            //             // for (size_t idet = 0; idet < n_det; idet++) {
            //             //     for (size_t isamp = 0; isamp < n_samp; isamp++) {
            //             //         std::cout << "NEST input quat " << idet << ": " << isamp << ": " << raw_quats[4*(idet * n_samp + isamp)] << ", " << raw_quats[4*(idet * n_samp + isamp)+1] << ", " << raw_quats[4*(idet * n_samp + isamp)+2] << ", " << raw_quats[4*(idet * n_samp + isamp)+3] << std::endl;
            //             //     }
            //             // }
            //         }
            //         #pragma acc parallel
            //         #pragma acc loop independent
            //         for (size_t idet = 0; idet < n_det; idet++) {
            //             int32_t q_indx = raw_quatindx[idet];
            //             int32_t p_indx = raw_pixindx[idet];
            //             #pragma acc loop independent
            //             for (size_t isamp = 0; isamp < n_samp; isamp++) {
            //                 const double zaxis[3] = {0.0, 0.0, 1.0};
            //                 double dir[3];
            //                 double z;
            //                 double rtz;
            //                 double phi;
            //                 int region;
            //                 size_t qoff = (q_indx * 4 * n_samp) + 4 * isamp;
            //                 size_t poff = p_indx * n_samp + isamp;
            //                 int64_t sub_map;

            //                 if ((raw_flags[isamp] & shared_flag_mask) == 0) {
            //                     // Good data
            //                     qa_rotate(&(raw_quats[qoff]), zaxis, dir);
            //                     hpix_vec2zphi(&hp, dir, &phi, &region, &z, &rtz);
            //                     hpix_zphi2nest(&hp, phi, region, z, rtz,
            //                                    &(raw_pixels[poff]));
            //                     sub_map = (int64_t)(raw_pixels[poff] / n_pix_submap);
            //                     raw_hsub[sub_map] = 1;
            //                 } else {
            //                     // Bad data
            //                     raw_pixels[poff] = -1;
            //                 }
            //             }
            //         }
            //         // if (fake_openacc()) {
            //         //     for (size_t idet = 0; idet < n_det; idet++) {
            //         //         for (size_t isamp = 0; isamp < n_samp; isamp++) {
            //         //             std::cout << "NEST output " << idet << ": " << isamp << ": " << raw_pixels[idet * n_samp + isamp] << std::endl;
            //         //         }
            //         //     }
            //         // }
            //     }
            // } else {
            //     #pragma \
            //     acc data copy(raw_hsub[:n_submap]) copyin(n_det, n_samp, shared_flag_mask, hp, n_pix_submap, raw_pixindx[:n_det], raw_quatindx[:n_det]) present(raw_pixels[:len_pixels], raw_flags[:len_flags], raw_quats[:len_quats])
            //     {
            //         if (fake_openacc()) {
            //             // Set all "present" data to point at the fake device pointers
            //             auto & fake = FakeMemPool::get();
            //             raw_pixels = (int64_t *)fake.device_ptr(raw_pixels);
            //             raw_flags = (uint8_t *)fake.device_ptr(raw_flags);
            //             raw_quats = (double *)fake.device_ptr(raw_quats);
            //             // for (size_t idet = 0; idet < n_det; idet++) {
            //             //     for (size_t isamp = 0; isamp < n_samp; isamp++) {
            //             //         std::cout << "RING input quat " << idet << ": " << isamp << ": " << raw_quats[4*(idet * n_samp + isamp)] << ", " << raw_quats[4*(idet * n_samp + isamp)+1] << ", " << raw_quats[4*(idet * n_samp + isamp)+2] << ", " << raw_quats[4*(idet * n_samp + isamp)+3] << std::endl;
            //             //     }
            //             // }
            //         }
            //         #pragma acc parallel
            //         #pragma acc loop independent
            //         for (size_t idet = 0; idet < n_det; idet++) {
            //             int32_t q_indx = raw_quatindx[idet];
            //             int32_t p_indx = raw_pixindx[idet];
            //             #pragma acc loop independent
            //             for (size_t isamp = 0; isamp < n_samp; isamp++) {
            //                 const double zaxis[3] = {0.0, 0.0, 1.0};
            //                 double dir[3];
            //                 double z;
            //                 double rtz;
            //                 double phi;
            //                 int region;
            //                 size_t qoff = (q_indx * 4 * n_samp) + 4 * isamp;
            //                 size_t poff = p_indx * n_samp + isamp;
            //                 int64_t sub_map;

            //                 if ((raw_flags[isamp] & shared_flag_mask) == 0) {
            //                     // Good data
            //                     qa_rotate(&(raw_quats[qoff]), zaxis, dir);
            //                     hpix_vec2zphi(&hp, dir, &phi, &region, &z, &rtz);
            //                     hpix_zphi2ring(&hp, phi, region, z, rtz,
            //                                    &(raw_pixels[poff]));
            //                     sub_map = (int64_t)(raw_pixels[poff] / n_pix_submap);
            //                     raw_hsub[sub_map] = 1;
            //                 } else {
            //                     // Bad data
            //                     raw_pixels[poff] = -1;
            //                 }
            //                 // std::cout << "hpixels " << isamp << ": "
            //                 // << raw_pixels[poff] << " sm = " << sub_map
            //                 // << std::endl;
            //             }
            //         }
            //         // if (fake_openacc()) {
            //         //     for (size_t idet = 0; idet < n_det; idet++) {
            //         //         for (size_t isamp = 0; isamp < n_samp; isamp++) {
            //         //             std::cout << "RING output " << idet << ": " << isamp << ": " << raw_pixels[idet * n_samp + isamp] << std::endl;
            //         //         }
            //         //     }
            //         // }
            //     }
            // }
            return;
        });
}
