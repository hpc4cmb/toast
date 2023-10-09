// Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// All compiled healpix functionality should be in this file, and it
// must be a single compilation unit.

#include <module.hpp>
#include <qarray.hpp>
#include <intervals.hpp>
#include <accelerator.hpp>
#include <cmath>

// 2 * Pi
const double TWOPI = 2.0 * M_PI;

// 2/3
const double TWOTHIRDS = 2.0 / 3.0;

const int64_t hpix_jr[] = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
const int64_t hpix_jp[] = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};

// Helper table initialization, performed on host.

void hpix_init_utab(int64_t * utab) {
    for (int64_t m = 0; m < 0x100; ++m) {
        utab[m] = (m & 0x1) | ((m & 0x2) << 1) | ((m & 0x4) << 2) |
                  ((m & 0x8) << 3) | ((m & 0x10) << 4) | ((m & 0x20) << 5) |
                  ((m & 0x40) << 6) | ((m & 0x80) << 7);
    }
    return;
}

void hpix_init_ctab(int64_t * ctab) {
    for (int64_t m = 0; m < 0x100; ++m) {
        ctab[m] = (m & 0x1) | ((m & 0x2) << 7) | ((m & 0x4) >> 1) |
                  ((m & 0x8) << 6) | ((m & 0x10) >> 2) | ((m & 0x20) << 5) |
                  ((m & 0x40) >> 3) | ((m & 0x80) << 4);
    }
    return;
}

#ifdef HAVE_OPENMP_TARGET
# pragma omp declare target
#endif // ifdef HAVE_OPENMP_TARGET

// Functions working with a single sample.  We use hpix_* prefix for these.

void hpix_qa_rotate(
    double const * q_in,
    double const * v_in,
    double * v_out
) {
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
                    (yw + xz) * v_in[2]) + v_in[0];

    v_out[1] = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] +
                    (yz - xw) * v_in[2]) + v_in[1];

    v_out[2] = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] +
                    (x2 + y2) * v_in[2]) + v_in[2];

    return;
}

int64_t hpix_xy2pix(int64_t const * utab, int64_t const & x, int64_t const & y) {
    return utab[x & 0xff] | (utab[(x >> 8) & 0xff] << 16) |
           (utab[(x >> 16) & 0xff] << 32) |
           (utab[(x >> 24) & 0xff] << 48) |
           (utab[y & 0xff] << 1) | (utab[(y >> 8) & 0xff] << 17) |
           (utab[(y >> 16) & 0xff] << 33) |
           (utab[(y >> 24) & 0xff] << 49);
}

void hpix_pix2xy(int64_t const * ctab, int64_t const & pix, int64_t & x, int64_t & y) {
    int64_t raw;
    raw = (pix & 0x5555ull) | ((pix & 0x55550000ull) >> 15) |
          ((pix & 0x555500000000ull) >> 16) |
          ((pix & 0x5555000000000000ull) >> 31);
    x = ctab[raw & 0xff] | (ctab[(raw >> 8) & 0xff] << 4) |
        (ctab[(raw >> 16) & 0xff] << 16) |
        (ctab[(raw >> 24) & 0xff] << 20);
    raw = ((pix & 0xaaaaull) >> 1) | ((pix & 0xaaaa0000ull) >> 16) |
          ((pix & 0xaaaa00000000ull) >> 17) |
          ((pix & 0xaaaa000000000000ull) >> 32);
    y = ctab[raw & 0xff] | (ctab[(raw >> 8) & 0xff] << 4) |
        (ctab[(raw >> 16) & 0xff] << 16) |
        (ctab[(raw >> 24) & 0xff] << 20);
    return;
}

void hpix_vec2zphi(
    double const * vec,
    double & phi,
    int & region,
    double & z,
    double & rtz
) {
    // region encodes BOTH the sign of Z and whether its
    // absolute value is greater than 2/3.
    z = vec[2];
    double za = fabs(z);
    int itemp = (z > 0.0) ? 1 : -1;
    region = (za <= TWOTHIRDS) ? itemp : itemp + itemp;
    rtz = sqrt(3.0 * (1.0 - za));
    phi = atan2(vec[1], vec[0]);
    return;
}

void hpix_zphi2nest(
    int64_t const & nside,
    int64_t const & factor,
    int64_t const * utab,
    double const & phi,
    int const & region,
    double const & z,
    double const & rtz,
    int64_t & pix
) {
    static const double eps = std::numeric_limits <double>::epsilon();
    double tol = 10.0 * eps;
    double phi_mod = ::fmod(phi, TWOPI);
    if ((phi_mod < tol) && (phi_mod > -tol)) {
        phi_mod = 0.0;
    }
    double tt = (phi_mod >= 0.0) ? phi_mod * M_2_PI : phi_mod * M_2_PI + 4.0;
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

    double dnside = static_cast <double> (nside);
    int64_t twonside = 2 * nside;
    double halfnside = 0.5 * dnside;
    double tqnside = 0.75 * dnside;
    int64_t nsideminusone = nside - 1;

    if ((region == 1) || (region == -1)) {
        temp1 = halfnside + dnside * tt;
        temp2 = tqnside * z;

        jp = (int64_t)(temp1 - temp2);
        jm = (int64_t)(temp1 + temp2);

        ifp = jp >> factor;
        ifm = jm >> factor;

        if (ifp == ifm) {
            face = (ifp == 4) ? (int64_t)4 : ifp + 4;
        } else if (ifp < ifm) {
            face = ifp;
        } else {
            face = ifm + 8;
        }

        x = jm & nsideminusone;
        y = nsideminusone - (jp & nsideminusone);
    } else {
        ntt = (int64_t)tt;

        tp = tt - (double)ntt;

        temp1 = dnside * rtz;

        jp = (int64_t)(tp * temp1);
        jm = (int64_t)((1.0 - tp) * temp1);

        if (jp >= nside) {
            jp = nsideminusone;
        }
        if (jm >= nside) {
            jm = nsideminusone;
        }

        if (z >= 0) {
            face = ntt;
            x = nsideminusone - jm;
            y = nsideminusone - jp;
        } else {
            face = ntt + 8;
            x = jp;
            y = jm;
        }
    }
    int64_t sipf = hpix_xy2pix(utab, x, y);
    pix = sipf + (face << (2 * factor));

    return;
}

void hpix_zphi2ring(
    int64_t const & nside,
    int64_t const & factor,
    double const & phi,
    int const & region,
    double const & z,
    double const & rtz,
    int64_t & pix
) {
    static const double eps = std::numeric_limits <double>::epsilon();
    double tol = 10.0 * eps;
    double phi_mod = ::fmod(phi, TWOPI);
    if ((phi_mod < tol) && (phi_mod > -tol)) {
        phi_mod = 0.0;
    }
    double tt = (phi_mod >= 0.0) ? phi_mod * M_2_PI : phi_mod * M_2_PI + 4.0;
    double tp;
    int64_t longpart;
    double temp1;
    double temp2;
    int64_t jp;
    int64_t jm;
    int64_t ip;
    int64_t ir;
    int64_t kshift;

    double dnside = static_cast <double> (nside);
    int64_t fournside = 4 * nside;
    double halfnside = 0.5 * dnside;
    double tqnside = 0.75 * dnside;
    int64_t nsideplusone = nside + 1;
    int64_t ncap = 2 * (nside * nside - nside);
    int64_t npix = 12 * nside * nside;

    if ((region == 1) || (region == -1)) {
        temp1 = halfnside + dnside * tt;
        temp2 = tqnside * z;

        jp = (int64_t)(temp1 - temp2);
        jm = (int64_t)(temp1 + temp2);

        ir = nsideplusone + jp - jm;
        kshift = 1 - (ir & 1);

        ip = (jp + jm - nside + kshift + 1) >> 1;
        ip = ip % fournside;

        pix = ncap + ((ir - 1) * fournside + ip);
    } else {
        tp = tt - floor(tt);

        temp1 = dnside * rtz;

        jp = (int64_t)(tp * temp1);
        jm = (int64_t)((1.0 - tp) * temp1);
        ir = jp + jm + 1;
        ip = (int64_t)(tt * (double)ir);
        longpart = (int64_t)(ip / (4 * ir));
        ip -= longpart;

        pix = (region > 0) ? (2 * ir * (ir - 1) + ip)
            : (npix - 2 * ir * (ir + 1) + ip);
    }

    return;
}

void hpix_ang2vec(double const & theta, double const & phi, double * vec) {
    double sintheta = ::sin(theta);
    vec[0] = sintheta * ::cos(phi);
    vec[1] = sintheta * ::sin(phi);
    vec[2] = ::cos(theta);
    return;
}

void hpix_vec2ang(
    double const * vec,
    double & theta,
    double & phi
) {
    static const double eps = std::numeric_limits <double>::epsilon();
    double norm = 1.0 / ::sqrt(
        vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]
    );
    theta = ::acos(vec[2] * norm);
    bool small_theta = (::fabs(theta) <= eps) ? true : false;
    bool big_theta = (::fabs(M_PI - theta) <= eps) ? true : false;
    double phitemp = ::atan2(vec[1], vec[0]);
    phi = (phitemp < 0) ? phitemp + TWOPI : phitemp;
    phi = (small_theta || big_theta) ? 0.0 : phi;
    return;
}

void hpix_theta2z(
    double const & theta,
    int & region,
    double & z,
    double & rtz
) {
    z = ::cos(theta);
    double za = ::fabs(z);
    int itemp = (z > 0.0) ? 1 : -1;
    region = (za <= TWOTHIRDS) ? itemp : itemp + itemp;
    double work = 3.0 * (1.0 - za);
    rtz = ::sqrt(work);
    return;
}

void hpix_ang2nest(
    int64_t const & nside,
    int64_t const & factor,
    int64_t const * utab,
    double const & theta,
    double const & phi,
    int64_t & pix
) {
    double z;
    double rtz;
    int region;
    hpix_theta2z(theta, region, z, rtz);
    hpix_zphi2nest(nside, factor, utab, phi, region, z, rtz, pix);
    return;
}

void hpix_ang2ring(
    int64_t const & nside,
    int64_t const & factor,
    int64_t const * utab,
    double const & theta,
    double const & phi,
    int64_t & pix
) {
    double z;
    double rtz;
    int region;
    hpix_theta2z(theta, region, z, rtz);
    hpix_zphi2ring(nside, factor, phi, region, z, rtz, pix);
    return;
}

void hpix_vec2nest(
    int64_t const & nside,
    int64_t const & factor,
    int64_t const * utab,
    double const * vec,
    int64_t & pix
) {
    double z;
    double phi;
    double rtz;
    int region;
    hpix_vec2zphi(vec, phi, region, z, rtz);
    hpix_zphi2nest(nside, factor, utab, phi, region, z, rtz, pix);
    return;
}

void hpix_vec2ring(
    int64_t const & nside,
    int64_t const & factor,
    int64_t const * utab,
    double const * vec,
    int64_t & pix
) {
    double z;
    double phi;
    double rtz;
    int region;
    hpix_vec2zphi(vec, phi, region, z, rtz);
    hpix_zphi2ring(nside, factor, phi, region, z, rtz, pix);
    return;
}

void hpix_ring2nest(
    int64_t const & nside,
    int64_t const & factor,
    int64_t const & npix,
    int64_t const & ncap,
    int64_t const * utab,
    int64_t const & ringpix,
    int64_t & nestpix
) {
    int64_t fc;
    int64_t x, y;
    int64_t nr;
    int64_t kshift;
    int64_t iring;
    int64_t iphi;
    int64_t tmp;
    int64_t ip;
    int64_t ire, irm;
    int64_t ifm, ifp;
    int64_t irt, ipt;
    if (ringpix < ncap) {
        iring = static_cast <int64_t> (
            0.5 * (
                1.0 + ::sqrt(static_cast <double> (1 + 2 * ringpix))
            )
        );
        iphi  = (ringpix + 1) - 2 * iring * (iring - 1);
        kshift = 0;
        nr = iring;
        fc = 0;
        tmp = iphi - 1;
        if (tmp >= (2 * iring)) {
            fc = 2;
            tmp -= 2 * iring;
        }
        if (tmp >= iring) {
            ++fc;
        }
    } else if (ringpix < (npix - ncap)) {
        ip = ringpix - ncap;
        iring = (ip >> (factor + 2)) + nside;
        iphi = (ip & (4 * nside - 1)) + 1;
        kshift = (iring + nside) & 1;
        nr = nside;
        ire = iring - nside + 1;
        irm = 2 * nside + 2 - ire;
        ifm = (iphi - (ire / 2) + nside - 1) >> factor;
        ifp = (iphi - (irm / 2) + nside - 1) >> factor;
        if (ifp == ifm) {
            // faces 4 to 7
            fc = (ifp == 4) ? 4 : ifp + 4;
        } else if (ifp < ifm) {
            // (half-)faces 0 to 3
            fc = ifp;
        } else {
            // (half-)faces 8 to 11
            fc = ifm + 8;
        }
    } else {
        ip = npix - ringpix;
        iring = static_cast <int64_t> (
            0.5 * (1.0 + ::sqrt(static_cast <double> (2 * ip - 1)))
        );
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
        kshift = 0;
        nr = iring;
        iring = 4 * nside - iring;
        fc = 8;
        tmp = iphi - 1;
        if (tmp >= (2 * nr)) {
            fc = 10;
            tmp -= 2 * nr;
        }
        if (tmp >= nr) {
            ++fc;
        }
    }
    irt = iring - hpix_jr[fc] * nside + 1;
    ipt = 2 * iphi - hpix_jp[fc] * nr - kshift - 1;
    if (ipt >= 2 * nside) {
        ipt -= 8 * nside;
    }
    x = (ipt - irt) >> 1;
    y = (-(ipt + irt)) >> 1;
    nestpix = hpix_xy2pix(utab, x, y);
    nestpix += (fc << (2 * factor));
    return;
}

void hpix_nest2ring(
    int64_t const & nside,
    int64_t const & factor,
    int64_t const & npix,
    int64_t const & ncap,
    int64_t const * ctab,
    int64_t const & nestpix,
    int64_t & ringpix
) {
    int64_t fc;
    int64_t x, y;
    int64_t jr;
    int64_t jp;
    int64_t nr;
    int64_t kshift;
    int64_t n_before;
    fc = nestpix >> (2 * factor);
    hpix_pix2xy(ctab, nestpix & (nside * nside - 1), x, y);
    jr = (hpix_jr[fc] * nside) - x - y - 1;
    if (jr < nside) {
        nr = jr;
        n_before = 2 * nr * (nr - 1);
        kshift = 0;
    } else if (jr > (3 * nside)) {
        nr = 4 * nside - jr;
        n_before = npix - 2 * (nr + 1) * nr;
        kshift = 0;
    } else {
        nr = nside;
        n_before = ncap + (jr - nside) * 4 * nside;
        kshift = (jr - nside) & 1;
    }
    jp = (hpix_jp[fc] * nr + x - y + 1 + kshift) / 2;
    if (jp > 4 * nside) {
        jp -= 4 * nside;
    } else {
        if (jp < 1) {
            jp += 4 * nside;
        }
    }
    ringpix = n_before + jp - 1;
    return;
}

void hpix_degrade_nest(
    int64_t const & degrade_levels,
    int64_t const & in_pix,
    int64_t & out_pix
) {
    out_pix = in_pix >> (2 * degrade_levels);
    return;
}

void hpix_degrade_ring(
    int64_t const & in_nside,
    int64_t const & in_factor,
    int64_t const & in_npix,
    int64_t const & in_ncap,
    int64_t const & out_nside,
    int64_t const & out_factor,
    int64_t const & out_npix,
    int64_t const & out_ncap,
    int64_t const * utab,
    int64_t const * ctab,
    int64_t const & degrade_levels,
    int64_t const & in_pix,
    int64_t & out_pix
) {
    int64_t in_nest;
    int64_t out_nest;
    hpix_ring2nest(in_nside, in_factor, in_npix, in_ncap, utab, in_pix, in_nest);
    hpix_degrade_nest(degrade_levels, in_nest, out_nest);
    hpix_nest2ring(out_nside, out_factor, out_npix, out_ncap, ctab, out_nest, out_pix);
    return;
}

void hpix_upgrade_nest(
    int64_t const & upgrade_levels,
    int64_t const & in_pix,
    int64_t & out_pix
) {
    out_pix = in_pix << (2 * upgrade_levels);
    return;
}

void hpix_upgrade_ring(
    int64_t const & in_nside,
    int64_t const & in_factor,
    int64_t const & in_npix,
    int64_t const & in_ncap,
    int64_t const & out_nside,
    int64_t const & out_factor,
    int64_t const & out_npix,
    int64_t const & out_ncap,
    int64_t const * utab,
    int64_t const * ctab,
    int64_t const & upgrade_levels,
    int64_t const & in_pix,
    int64_t & out_pix
) {
    int64_t in_nest;
    int64_t out_nest;
    hpix_ring2nest(in_nside, in_factor, in_npix, in_ncap, utab, in_pix, in_nest);
    hpix_upgrade_nest(upgrade_levels, in_nest, out_nest);
    hpix_nest2ring(out_nside, out_factor, out_npix, out_ncap, ctab, out_nest, out_pix);
    return;
}

void pixels_healpix_nest_inner(
    int64_t nside,
    int64_t factor,
    int64_t * utab,
    int32_t const * quat_index,
    int32_t const * pixel_index,
    double const * quats,
    uint8_t const * flags,
    uint8_t * hsub,
    int64_t * pixels,
    int64_t n_pix_submap,
    int64_t isamp,
    int64_t n_samp,
    int64_t idet,
    uint8_t mask,
    bool use_flags
) {
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

    hpix_qa_rotate(&(quats[qoff]), zaxis, dir);
    hpix_vec2zphi(dir, phi, region, z, rtz);
    hpix_zphi2nest(nside, factor, utab, phi, region, z, rtz, pixels[poff]);
    if (use_flags && ((flags[isamp] & mask) != 0)) {
        pixels[poff] = -1;
    } else {
        sub_map = (int64_t)(pixels[poff] / n_pix_submap);
        hsub[sub_map] = 1;
    }

    return;
}

void pixels_healpix_ring_inner(
    int64_t nside,
    int64_t factor,
    int32_t const * quat_index,
    int32_t const * pixel_index,
    double const * quats,
    uint8_t const * flags,
    uint8_t * hsub,
    int64_t * pixels,
    int64_t n_pix_submap,
    int64_t isamp,
    int64_t n_samp,
    int64_t idet,
    uint8_t mask,
    bool use_flags) {
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

    hpix_qa_rotate(&(quats[qoff]), zaxis, dir);
    hpix_vec2zphi(dir, phi, region, z, rtz);
    hpix_zphi2ring(nside, factor, phi, region, z, rtz, pixels[poff]);
    if (use_flags && ((flags[isamp] & mask) != 0)) {
        pixels[poff] = -1;
    } else {
        sub_map = (int64_t)(pixels[poff] / n_pix_submap);
        hsub[sub_map] = 1;
    }

    return;
}

#ifdef HAVE_OPENMP_TARGET
# pragma omp end declare target
#endif // ifdef HAVE_OPENMP_TARGET

void init_ops_pixels_healpix(py::module & m) {
    m.def(
        "healpix_ang2vec", [](
            py::buffer theta,
            py::buffer phi,
            py::buffer vec
        ) {
            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(2);

            double * raw_theta = extract_buffer <double> (
                theta, "theta", 1, temp_shape, {-1}
            );
            int64_t n_samp = temp_shape[0];
            double * raw_phi = extract_buffer <double> (
                phi, "phi", 1, temp_shape, {n_samp}
            );
            double * raw_vec = extract_buffer <double> (
                vec, "vec", 2, temp_shape, {n_samp, 3}
            );

            #pragma omp parallel for default(shared) schedule(static)
            for (int64_t isamp = 0; isamp < n_samp; isamp++) {
                hpix_ang2vec(raw_theta[isamp], raw_phi[isamp], &(raw_vec[3 * isamp]));
            }
            return;
        }
    );

    m.def(
        "healpix_vec2ang", [](
            py::buffer vec,
            py::buffer theta,
            py::buffer phi
        ) {
            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(2);

            double * raw_vec = extract_buffer <double> (
                vec, "vec", 2, temp_shape, {-1, 3}
            );
            int64_t n_samp = temp_shape[0];
            double * raw_theta = extract_buffer <double> (
                theta, "theta", 1, temp_shape, {n_samp}
            );
            double * raw_phi = extract_buffer <double> (
                phi, "phi", 1, temp_shape, {n_samp}
            );

            #pragma omp parallel for default(shared) schedule(static)
            for (int64_t isamp = 0; isamp < n_samp; isamp++) {
                hpix_vec2ang(&(raw_vec[3 * isamp]), raw_theta[isamp], raw_phi[isamp]);
            }
            return;
        }
    );

    m.def(
        "healpix_ang2nest", [](
            int64_t nside,
            py::buffer theta,
            py::buffer phi,
            py::buffer pix
        ) {
            static int64_t utab[0x100];
            hpix_init_utab(utab);
            int64_t factor = 0;
            while (nside != (1ll << factor)) {
                ++factor;
            }

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(2);

            double * raw_theta = extract_buffer <double> (
                theta, "theta", 1, temp_shape, {-1}
            );
            int64_t n_samp = temp_shape[0];
            double * raw_phi = extract_buffer <double> (
                phi, "phi", 1, temp_shape, {n_samp}
            );
            int64_t * raw_pix = extract_buffer <int64_t> (
                pix, "pix", 1, temp_shape, {n_samp}
            );

            #pragma omp parallel for default(shared) schedule(static)
            for (int64_t isamp = 0; isamp < n_samp; isamp++) {
                hpix_ang2nest(
                    nside,
                    factor,
                    utab,
                    raw_theta[isamp],
                    raw_phi[isamp],
                    raw_pix[isamp]
                );
            }
            return;
        }
    );

    m.def(
        "healpix_ang2ring", [](
            int64_t nside,
            py::buffer theta,
            py::buffer phi,
            py::buffer pix
        ) {
            static int64_t utab[0x100];
            hpix_init_utab(utab);
            int64_t factor = 0;
            while (nside != (1ll << factor)) {
                ++factor;
            }

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(2);

            double * raw_theta = extract_buffer <double> (
                theta, "theta", 1, temp_shape, {-1}
            );
            int64_t n_samp = temp_shape[0];
            double * raw_phi = extract_buffer <double> (
                phi, "phi", 1, temp_shape, {n_samp}
            );
            int64_t * raw_pix = extract_buffer <int64_t> (
                pix, "pix", 1, temp_shape, {n_samp}
            );

            #pragma omp parallel for default(shared) schedule(static)
            for (int64_t isamp = 0; isamp < n_samp; isamp++) {
                hpix_ang2ring(
                    nside,
                    factor,
                    utab,
                    raw_theta[isamp],
                    raw_phi[isamp],
                    raw_pix[isamp]
                );
            }
            return;
        }
    );

    m.def(
        "healpix_vec2nest", [](
            int64_t nside,
            py::buffer vec,
            py::buffer pix
        ) {
            static int64_t utab[0x100];
            hpix_init_utab(utab);
            int64_t factor = 0;
            while (nside != (1ll << factor)) {
                ++factor;
            }

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(2);

            double * raw_vec = extract_buffer <double> (
                vec, "vec", 2, temp_shape, {-1, 3}
            );
            int64_t n_samp = temp_shape[0];
            int64_t * raw_pix = extract_buffer <int64_t> (
                pix, "pix", 1, temp_shape, {n_samp}
            );

            #pragma omp parallel for default(shared) schedule(static)
            for (int64_t isamp = 0; isamp < n_samp; isamp++) {
                hpix_vec2nest(
                    nside,
                    factor,
                    utab,
                    &(raw_vec[3 * isamp]),
                    raw_pix[isamp]
                );
            }
            return;
        }
    );

    m.def(
        "healpix_vec2ring", [](
            int64_t nside,
            py::buffer vec,
            py::buffer pix
        ) {
            static int64_t utab[0x100];
            hpix_init_utab(utab);
            int64_t factor = 0;
            while (nside != (1ll << factor)) {
                ++factor;
            }

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(2);

            double * raw_vec = extract_buffer <double> (
                vec, "vec", 2, temp_shape, {-1, 3}
            );
            int64_t n_samp = temp_shape[0];
            int64_t * raw_pix = extract_buffer <int64_t> (
                pix, "pix", 1, temp_shape, {n_samp}
            );

            #pragma omp parallel for default(shared) schedule(static)
            for (int64_t isamp = 0; isamp < n_samp; isamp++) {
                hpix_vec2ring(
                    nside,
                    factor,
                    utab,
                    &(raw_vec[3 * isamp]),
                    raw_pix[isamp]
                );
            }
            return;
        }
    );

    m.def(
        "healpix_ring2nest", [](
            int64_t nside,
            py::buffer ring_pix,
            py::buffer nest_pix
        ) {
            static int64_t utab[0x100];
            hpix_init_utab(utab);
            int64_t npix = 12 * nside * nside;
            int64_t ncap = 2 * (nside * nside - nside);
            int64_t factor = 0;
            while (nside != (1ll << factor)) {
                ++factor;
            }

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(2);

            int64_t * raw_ring_pix = extract_buffer <int64_t> (
                ring_pix, "ring_pix", 1, temp_shape, {-1}
            );
            int64_t n_samp = temp_shape[0];
            int64_t * raw_nest_pix = extract_buffer <int64_t> (
                nest_pix, "nest_pix", 1, temp_shape, {n_samp}
            );

            #pragma omp parallel for default(shared) schedule(static)
            for (int64_t isamp = 0; isamp < n_samp; isamp++) {
                hpix_ring2nest(
                    nside,
                    factor,
                    npix,
                    ncap,
                    utab,
                    raw_ring_pix[isamp],
                    raw_nest_pix[isamp]
                );
            }
            return;
        }
    );

    m.def(
        "healpix_nest2ring", [](
            int64_t nside,
            py::buffer nest_pix,
            py::buffer ring_pix
        ) {
            static int64_t ctab[0x100];
            hpix_init_ctab(ctab);
            int64_t npix = 12 * nside * nside;
            int64_t ncap = 2 * (nside * nside - nside);
            int64_t factor = 0;
            while (nside != (1ll << factor)) {
                ++factor;
            }

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(2);

            int64_t * raw_nest_pix = extract_buffer <int64_t> (
                nest_pix, "nest_pix", 1, temp_shape, {-1}
            );
            int64_t n_samp = temp_shape[0];
            int64_t * raw_ring_pix = extract_buffer <int64_t> (
                ring_pix, "ring_pix", 1, temp_shape, {n_samp}
            );

            #pragma omp parallel for default(shared) schedule(static)
            for (int64_t isamp = 0; isamp < n_samp; isamp++) {
                hpix_nest2ring(
                    nside,
                    factor,
                    npix,
                    ncap,
                    ctab,
                    raw_nest_pix[isamp],
                    raw_ring_pix[isamp]
                );
            }
            return;
        }
    );

    m.def(
        "healpix_degrade_ring", [](
            int64_t in_nside,
            int64_t degrade_factor,
            py::buffer in_pix,
            py::buffer out_pix
        ) {
            static int64_t utab[0x100];
            hpix_init_utab(utab);
            static int64_t ctab[0x100];
            hpix_init_ctab(ctab);

            int64_t in_npix = 12 * in_nside * in_nside;
            int64_t in_ncap = 2 * (in_nside * in_nside - in_nside);
            int64_t in_factor = 0;
            while (in_nside != (1ll << in_factor)) {
                ++in_factor;
            }
            int64_t out_nside = in_nside >> degrade_factor;
            int64_t out_npix = 12 * out_nside * out_nside;
            int64_t out_ncap = 2 * (out_nside * out_nside - out_nside);
            int64_t out_factor = 0;
            while (out_nside != (1ll << out_factor)) {
                ++out_factor;
            }

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(2);

            int64_t * raw_in_pix = extract_buffer <int64_t> (
                in_pix, "in_pix", 1, temp_shape, {-1}
            );
            int64_t n_samp = temp_shape[0];
            int64_t * raw_out_pix = extract_buffer <int64_t> (
                out_pix, "out_pix", 1, temp_shape, {n_samp}
            );

            #pragma omp parallel for default(shared) schedule(static)
            for (int64_t isamp = 0; isamp < n_samp; isamp++) {
                hpix_degrade_ring(
                    in_nside,
                    in_factor,
                    in_npix,
                    in_ncap,
                    out_nside,
                    out_factor,
                    out_npix,
                    out_ncap,
                    utab,
                    ctab,
                    degrade_factor,
                    raw_in_pix[isamp],
                    raw_out_pix[isamp]
                );
            }
            return;
        }
    );

    m.def(
        "healpix_degrade_nest", [](
            int64_t in_nside,
            int64_t degrade_factor,
            py::buffer in_pix,
            py::buffer out_pix
        ) {
            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(2);

            int64_t * raw_in_pix = extract_buffer <int64_t> (
                in_pix, "in_pix", 1, temp_shape, {-1}
            );
            int64_t n_samp = temp_shape[0];
            int64_t * raw_out_pix = extract_buffer <int64_t> (
                out_pix, "out_pix", 1, temp_shape, {n_samp}
            );

            #pragma omp parallel for default(shared) schedule(static)
            for (int64_t isamp = 0; isamp < n_samp; isamp++) {
                hpix_degrade_nest(
                    degrade_factor,
                    raw_in_pix[isamp],
                    raw_out_pix[isamp]
                );
            }
            return;
        }
    );

    m.def(
        "healpix_upgrade_ring", [](
            int64_t in_nside,
            int64_t upgrade_factor,
            py::buffer in_pix,
            py::buffer out_pix
        ) {
            static int64_t utab[0x100];
            hpix_init_utab(utab);
            static int64_t ctab[0x100];
            hpix_init_ctab(ctab);

            int64_t in_npix = 12 * in_nside * in_nside;
            int64_t in_ncap = 2 * (in_nside * in_nside - in_nside);
            int64_t in_factor = 0;
            while (in_nside != (1ll << in_factor)) {
                ++in_factor;
            }
            int64_t out_nside = in_nside << upgrade_factor;
            int64_t out_npix = 12 * out_nside * out_nside;
            int64_t out_ncap = 2 * (out_nside * out_nside - out_nside);
            int64_t out_factor = 0;
            while (out_nside != (1ll << out_factor)) {
                ++out_factor;
            }

            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(2);

            int64_t * raw_in_pix = extract_buffer <int64_t> (
                in_pix, "in_pix", 1, temp_shape, {-1}
            );
            int64_t n_samp = temp_shape[0];
            int64_t * raw_out_pix = extract_buffer <int64_t> (
                out_pix, "out_pix", 1, temp_shape, {n_samp}
            );

            #pragma omp parallel for default(shared) schedule(static)
            for (int64_t isamp = 0; isamp < n_samp; isamp++) {
                hpix_upgrade_ring(
                    in_nside,
                    in_factor,
                    in_npix,
                    in_ncap,
                    out_nside,
                    out_factor,
                    out_npix,
                    out_ncap,
                    utab,
                    ctab,
                    upgrade_factor,
                    raw_in_pix[isamp],
                    raw_out_pix[isamp]
                );
            }
            return;
        }
    );

    m.def(
        "healpix_upgrade_nest", [](
            int64_t in_nside,
            int64_t upgrade_factor,
            py::buffer in_pix,
            py::buffer out_pix
        ) {
            // This is used to return the actual shape of each buffer
            std::vector <int64_t> temp_shape(2);

            int64_t * raw_in_pix = extract_buffer <int64_t> (
                in_pix, "in_pix", 1, temp_shape, {-1}
            );
            int64_t n_samp = temp_shape[0];
            int64_t * raw_out_pix = extract_buffer <int64_t> (
                out_pix, "out_pix", 1, temp_shape, {n_samp}
            );

            #pragma omp parallel for default(shared) schedule(static)
            for (int64_t isamp = 0; isamp < n_samp; isamp++) {
                hpix_upgrade_nest(
                    upgrade_factor,
                    raw_in_pix[isamp],
                    raw_out_pix[isamp]
                );
            }
            return;
        }
    );

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
            bool use_accel
        ) {
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

            static int64_t utab[0x100];
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
                size_t utab_bytes = 0x100 * sizeof(int64_t);
                int present_utab = omgr.present(utab, utab_bytes);
                if (present_utab == 0) {
                    void * vptr = omgr.create((void *)utab, utab_bytes);
                    omgr.update_device((void *)utab, utab_bytes);
                }

                int64_t * dev_utab = omgr.device_ptr(
                    utab);

                // Calculate the maximum interval size on the CPU
                int64_t max_interval_size = 0;
                for (int64_t iview = 0; iview < n_view; iview++) {
                    int64_t interval_size = raw_intervals[iview].last -
                                            raw_intervals[iview].first + 1;
                    if (interval_size > max_interval_size) {
                        max_interval_size = interval_size;
                    }
                }

                # pragma omp target data map(    \
                to : raw_pixel_index[0 : n_det], \
                raw_quat_index[0 : n_det],       \
                n_pix_submap,                    \
                nside,                           \
                factor,                          \
                nest,                            \
                n_view,                          \
                n_det,                           \
                n_samp,                          \
                shared_flag_mask,                \
                use_flags                        \
                )                                \
                map(tofrom : raw_hsub[0 : n_submap])
                {
                    if (nest) {
                        # pragma omp target teams distribute parallel for collapse(3) \
                        schedule(static,1)                                            \
                        is_device_ptr(                                                \
                        dev_pixels,                                                   \
                        dev_quats,                                                    \
                        dev_flags,                                                    \
                        dev_intervals,                                                \
                        dev_utab                                                      \
                        )
                        for (int64_t idet = 0; idet < n_det; idet++) {
                            for (int64_t iview = 0; iview < n_view; iview++) {
                                for (int64_t isamp = 0; isamp < max_interval_size;
                                     isamp++) {
                                    // Adjust for the actual start of the interval
                                    int64_t adjusted_isamp = isamp + dev_intervals[iview].first;

                                    // Check if the value is out of range for the
                                    // current interval
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
                        # pragma omp target teams distribute parallel for collapse(3) \
                        schedule(static,1)                                            \
                        is_device_ptr(                                                \
                        dev_pixels,                                                   \
                        dev_quats,                                                    \
                        dev_flags,                                                    \
                        dev_intervals,                                                \
                        dev_utab                                                      \
                        )
                        for (int64_t idet = 0; idet < n_det; idet++) {
                            for (int64_t iview = 0; iview < n_view; iview++) {
                                for (int64_t isamp = 0; isamp < max_interval_size;
                                     isamp++) {
                                    // Adjust for the actual start of the interval
                                    int64_t adjusted_isamp = isamp + dev_intervals[iview].first;

                                    // Check if the value is out of range for the
                                    // current interval
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
                            #pragma omp parallel for default(shared) schedule(static)
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
                            #pragma omp parallel for default(shared) schedule(static)
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
            return;
        });
}
