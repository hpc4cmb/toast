// Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// 2/3
#define TWOTHIRDS 0.66666666666666666667

// Double precision machine epsilon
#define EPS 2.220446e-16

// Internal functions working with a single sample.  We use hpix_* prefix for these.

double hpix_fmod(double in, double mod) {
    double div = in / mod;
    return mod * (div - (double)((long)div));
}

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

long hpix_xy2pix(__global long const * utab, long x, long y) {
    return utab[x & 0xff] | (utab[(x >> 8) & 0xff] << 16) |
           (utab[(x >> 16) & 0xff] << 32) |
           (utab[(x >> 24) & 0xff] << 48) |
           (utab[y & 0xff] << 1) | (utab[(y >> 8) & 0xff] << 17) |
           (utab[(y >> 16) & 0xff] << 33) |
           (utab[(y >> 24) & 0xff] << 49);
}

void hpix_pix2xy(__global long const * ctab, long pix, long * x, long * y) {
    long raw;
    raw = (pix & 0x5555ull) | ((pix & 0x55550000ull) >> 15) |
          ((pix & 0x555500000000ull) >> 16) |
          ((pix & 0x5555000000000000ull) >> 31);
    (*x) = ctab[raw & 0xff] | (ctab[(raw >> 8) & 0xff] << 4) |
        (ctab[(raw >> 16) & 0xff] << 16) |
        (ctab[(raw >> 24) & 0xff] << 20);
    raw = ((pix & 0xaaaaull) >> 1) | ((pix & 0xaaaa0000ull) >> 16) |
          ((pix & 0xaaaa00000000ull) >> 17) |
          ((pix & 0xaaaa000000000000ull) >> 32);
    (*y) = ctab[raw & 0xff] | (ctab[(raw >> 8) & 0xff] << 4) |
        (ctab[(raw >> 16) & 0xff] << 16) |
        (ctab[(raw >> 24) & 0xff] << 20);
    return;
}

void hpix_vec2zphi(
    double const * vec,
    double * phi,
    int * region,
    double * z,
    double * rtz
) {
    // region encodes BOTH the sign of Z and whether its
    // absolute value is greater than 2/3.
    (*z) = vec[2];
    double za = fabs((*z));
    int itemp = ((*z) > 0.0) ? 1 : -1;
    (*region) = (za <= TWOTHIRDS) ? itemp : itemp + itemp;
    (*rtz) = sqrt(3.0 * (1.0 - za));
    (*phi) = atan2(vec[1], vec[0]);
    return;
}

long hpix_zphi2nest(
    long nside,
    long factor,
    __global long const * utab,
    double phi,
    int region,
    double z,
    double rtz
) {
    double tol = 10.0 * EPS;
    double phi_mod = hpix_fmod(phi, 2 * M_PI);
    if ((phi_mod < tol) && (phi_mod > -tol)) {
        phi_mod = 0.0;
    }
    double tt = (phi_mod >= 0.0) ? phi_mod * M_2_PI : phi_mod * M_2_PI + 4.0;
    long x;
    long y;
    double temp1;
    double temp2;
    long jp;
    long jm;
    long ifp;
    long ifm;
    long face;
    long ntt;
    double tp;

    double dnside = (double)nside;
    double halfnside = 0.5 * dnside;
    double tqnside = 0.75 * dnside;
    long nsideminusone = nside - 1;

    if ((region == 1) || (region == -1)) {
        temp1 = halfnside + dnside * tt;
        temp2 = tqnside * z;

        jp = (long)(temp1 - temp2);
        jm = (long)(temp1 + temp2);

        ifp = jp >> factor;
        ifm = jm >> factor;

        if (ifp == ifm) {
            face = (ifp == 4) ? (long)4 : ifp + 4;
        } else if (ifp < ifm) {
            face = ifp;
        } else {
            face = ifm + 8;
        }

        x = jm & nsideminusone;
        y = nsideminusone - (jp & nsideminusone);
    } else {
        ntt = (long)tt;

        tp = tt - (double)ntt;

        temp1 = dnside * rtz;

        jp = (long)(tp * temp1);
        jm = (long)((1.0 - tp) * temp1);

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
    long sipf = hpix_xy2pix(utab, x, y);
    long pix = sipf + (face << (2 * factor));
    return pix;
}

long hpix_zphi2ring(
    long nside,
    long factor,
    double phi,
    int region,
    double z,
    double rtz
) {
    double tol = 10.0 * EPS;
    double phi_mod = hpix_fmod(phi, 2 * M_PI);
    if ((phi_mod < tol) && (phi_mod > -tol)) {
        phi_mod = 0.0;
    }
    double tt = (phi_mod >= 0.0) ? phi_mod * M_2_PI : phi_mod * M_2_PI + 4.0;
    double tp;
    long longpart;
    double temp1;
    double temp2;
    long jp;
    long jm;
    long ip;
    long ir;
    long kshift;
    long pix;

    double dnside = (double)nside;
    long fournside = 4 * nside;
    double halfnside = 0.5 * dnside;
    double tqnside = 0.75 * dnside;
    long nsideplusone = nside + 1;
    long ncap = 2 * (nside * nside - nside);
    long npix = 12 * nside * nside;

    if ((region == 1) || (region == -1)) {
        temp1 = halfnside + dnside * tt;
        temp2 = tqnside * z;

        jp = (long)(temp1 - temp2);
        jm = (long)(temp1 + temp2);

        ir = nsideplusone + jp - jm;
        kshift = 1 - (ir & 1);

        ip = (jp + jm - nside + kshift + 1) >> 1;
        ip = ip % fournside;

        pix = ncap + ((ir - 1) * fournside + ip);
    } else {
        tp = tt - floor(tt);

        temp1 = dnside * rtz;

        jp = (long)(tp * temp1);
        jm = (long)((1.0 - tp) * temp1);
        ir = jp + jm + 1;
        ip = (long)(tt * (double)ir);
        longpart = (long)(ip / (4 * ir));
        ip -= longpart;

        pix = (region > 0) ? (2 * ir * (ir - 1) + ip)
            : (npix - 2 * ir * (ir + 1) + ip);
    }
    return pix;
}

void hpix_ang2vec(double theta, double phi, double * vec) {
    double sintheta = sin(theta);
    vec[0] = sintheta * cos(phi);
    vec[1] = sintheta * sin(phi);
    vec[2] = cos(theta);
    return;
}

void hpix_vec2ang(
    double const * vec,
    double * theta,
    double * phi
) {
    double norm = 1.0 / sqrt(
        vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]
    );
    (*theta) = acos(vec[2] * norm);
    unsigned char small_theta = (fabs((*theta)) <= EPS) ? 1 : 0;
    unsigned char big_theta = (fabs(M_PI - (*theta)) <= EPS) ? 1 : 0;
    double phitemp = atan2(vec[1], vec[0]);
    (*phi) = (phitemp < 0) ? phitemp + 2 * M_PI : phitemp;
    (*phi) = (small_theta || big_theta) ? 0.0 : (*phi);
    return;
}

void hpix_theta2z(
    double theta,
    int * region,
    double * z,
    double * rtz
) {
    (*z) = cos(theta);
    double za = fabs((*z));
    int itemp = ((*z) > 0.0) ? 1 : -1;
    (*region) = (za <= TWOTHIRDS) ? itemp : itemp + itemp;
    double work = 3.0 * (1.0 - za);
    (*rtz) = sqrt(work);
    return;
}

long hpix_ang2nest(
    long nside,
    long factor,
    __global long const * utab,
    double theta,
    double phi
) {
    double z;
    double rtz;
    int region;
    long pix;
    hpix_theta2z(theta, &region, &z, &rtz);
    return hpix_zphi2nest(nside, factor, utab, phi, region, z, rtz);
}

long hpix_ang2ring(
    long nside,
    long factor,
    __global long const * utab,
    double theta,
    double phi
) {
    double z;
    double rtz;
    int region;
    hpix_theta2z(theta, &region, &z, &rtz);
    return hpix_zphi2ring(nside, factor, phi, region, z, rtz);
}

long hpix_vec2nest(
    long nside,
    long factor,
    __global long const * utab,
    double const * vec
) {
    double z;
    double phi;
    double rtz;
    int region;
    hpix_vec2zphi(vec, &phi, &region, &z, &rtz);
    return hpix_zphi2nest(nside, factor, utab, phi, region, z, rtz);
}

long hpix_vec2ring(
    long nside,
    long factor,
    __global long const * utab,
    double const * vec
) {
    double z;
    double phi;
    double rtz;
    int region;
    hpix_vec2zphi(vec, &phi, &region, &z, &rtz);
    return hpix_zphi2ring(nside, factor, phi, region, z, rtz);
}

long hpix_ring2nest(
    long nside,
    long factor,
    long npix,
    long ncap,
    __global long const * utab,
    long ringpix
) {
    long fc;
    long x, y;
    long nr;
    long kshift;
    long iring;
    long iphi;
    long tmp;
    long ip;
    long ire, irm;
    long ifm, ifp;
    long irt, ipt;
    const long hpix_jr[] = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    const long hpix_jp[] = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};
    long nestpix;

    if (ringpix < ncap) {
        iring = (long)(0.5 * (1.0 + sqrt((double)(1 + 2 * ringpix))));
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
        iring = (long)(0.5 * (1.0 + sqrt((double)(2 * ip - 1))));
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
    return nestpix;
}

long hpix_nest2ring(
    long nside,
    long factor,
    long npix,
    long ncap,
    __global long const * ctab,
    long nestpix
) {
    long fc;
    long x, y;
    long jr;
    long jp;
    long nr;
    long kshift;
    long n_before;
    const long hpix_jr[] = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    const long hpix_jp[] = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};
    long ringpix;

    fc = nestpix >> (2 * factor);
    hpix_pix2xy(ctab, nestpix & (nside * nside - 1), &x, &y);
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
    return ringpix;
}

long hpix_degrade_nest(
    long degrade_levels,
    long in_pix
) {
    return in_pix >> (2 * degrade_levels);
}

long hpix_degrade_ring(
    long in_nside,
    long in_factor,
    long in_npix,
    long in_ncap,
    long out_nside,
    long out_factor,
    long out_npix,
    long out_ncap,
    __global long const * utab,
    __global long const * ctab,
    long degrade_levels,
    long in_pix
) {
    long in_nest = hpix_ring2nest(in_nside, in_factor, in_npix, in_ncap, utab, in_pix);
    long out_nest = hpix_degrade_nest(degrade_levels, in_nest);
    return hpix_nest2ring(out_nside, out_factor, out_npix, out_ncap, ctab, out_nest);
}

long hpix_upgrade_nest(
    long upgrade_levels,
    long in_pix
) {
    return in_pix << (2 * upgrade_levels);
}

long hpix_upgrade_ring(
    long in_nside,
    long in_factor,
    long in_npix,
    long in_ncap,
    long out_nside,
    long out_factor,
    long out_npix,
    long out_ncap,
    __global long const * utab,
    __global long const * ctab,
    long upgrade_levels,
    long in_pix
) {
    long in_nest = hpix_ring2nest(in_nside, in_factor, in_npix, in_ncap, utab, in_pix);
    long out_nest = hpix_upgrade_nest(upgrade_levels, in_nest);
    return hpix_nest2ring(out_nside, out_factor, out_npix, out_ncap, ctab, out_nest);
}

// Kernels

__kernel void pixels_healpix_nest(
    int n_det,
    long n_sample,
    long first_sample,
    long nside,
    long factor,
    long n_pix_submap,
    __global long const * utab,
    __global int const * quat_index,
    __global double const * quats,
    __global int const * pixel_index,
    __global long * pixels,
    __global unsigned char * hsub,
    __global unsigned char const * shared_flags,
    unsigned char shared_flag_mask,
    unsigned char use_flags,
    unsigned char compute_submaps
) {
    // Get the global index of this work element
    int idet = get_global_id(0);
    long isamp = first_sample + get_global_id(1);

    const double zaxis[3] = {0.0, 0.0, 1.0};
    int p_indx = pixel_index[idet];
    int q_indx = quat_index[idet];
    double dir[3];
    double z;
    double rtz;
    double phi;
    int region;
    size_t qoff = (q_indx * 4 * n_sample) + 4 * isamp;
    size_t poff = p_indx * n_sample + isamp;
    long sub_map;

    // Copy to private variable in order to pass to subroutines.
    double temp_quat[4];
    temp_quat[0] = quats[qoff];
    temp_quat[1] = quats[qoff + 1];
    temp_quat[2] = quats[qoff + 2];
    temp_quat[3] = quats[qoff + 3];

    hpix_qa_rotate(temp_quat, zaxis, dir);
    hpix_vec2zphi(dir, &phi, &region, &z, &rtz);
    pixels[poff] = hpix_zphi2nest(nside, factor, utab, phi, region, z, rtz);
    if (use_flags && ((shared_flags[isamp] & shared_flag_mask) != 0)) {
        pixels[poff] = -1;
    } else {
        if (compute_submaps) {
            sub_map = (long)(pixels[poff] / n_pix_submap);
            hsub[sub_map] = 1;
        }
    }

    return;
}

// Note:  Although utab is not needed for ang to ring pix,
// we keep it in the argument list to simplify the calling
// code.

__kernel void pixels_healpix_ring(
    int n_det,
    long n_sample,
    long first_sample,
    long nside,
    long factor,
    long n_pix_submap,
    __global long const * utab,
    __global int const * quat_index,
    __global double const * quats,
    __global int const * pixel_index,
    __global long * pixels,
    __global unsigned char * hsub,
    __global unsigned char const * shared_flags,
    unsigned char shared_flag_mask,
    unsigned char use_flags,
    unsigned char compute_submaps
) {
    // Get the global index of this work element
    int idet = get_global_id(0);
    long isamp = first_sample + get_global_id(1);

    const double zaxis[3] = {0.0, 0.0, 1.0};
    int p_indx = pixel_index[idet];
    int q_indx = quat_index[idet];
    double dir[3];
    double z;
    double rtz;
    double phi;
    int region;
    size_t qoff = (q_indx * 4 * n_sample) + 4 * isamp;
    size_t poff = p_indx * n_sample + isamp;
    long sub_map;

    // Copy to private variable in order to pass to subroutines.
    double temp_quat[4];
    temp_quat[0] = quats[qoff];
    temp_quat[1] = quats[qoff + 1];
    temp_quat[2] = quats[qoff + 2];
    temp_quat[3] = quats[qoff + 3];

    hpix_qa_rotate(temp_quat, zaxis, dir);
    hpix_vec2zphi(dir, &phi, &region, &z, &rtz);
    pixels[poff] = hpix_zphi2ring(nside, factor, phi, region, z, rtz);
    if (use_flags && ((shared_flags[isamp] & shared_flag_mask) != 0)) {
        pixels[poff] = -1;
    } else {
        if (compute_submaps) {
            sub_map = (long)(pixels[poff] / n_pix_submap);
            hsub[sub_map] = 1;
        }
    }

    return;
}
