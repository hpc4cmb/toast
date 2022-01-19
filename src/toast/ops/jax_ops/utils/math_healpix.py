# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import jax
import jax.numpy as jnp

# -------------------------------------------------------------------------------------------------
# JAX

# -------------------------------------------------------------------------------------------------
# NUMPY

TWOINVPI = 0.63661977236758134308


def zphi2nest(hpix, phi, region, z, rtz, pix):
    """
    Args:
        hpix (HealpixPixels):  The healpix projection object.
        phi (array, double) of size n
        region (array, int) of size n
        z (array, double) of size n
        rtz (array, double) of size n
        pix (array, int) of size n

    Returns:
        None (the results are put in pix)
    """
    # machine epsilon
    eps = np.finfo(np.float).eps

    # TODO that loop is a prime target for a vmap in jax
    n = pix.size
    for i in range(n):
        ph = phi[i]
        if np.abs(ph) < eps:
            ph = 0.0

        tt = ph * TWOINVPI
        if ph < 0.0:
            tt += 4.0

        if (np.abs(region[i]) == 1):
            temp1 = hpix.halfnside_ + hpix.dnside_ * tt
            temp2 = hpix.tqnside_ * z[i]

            jp = np.int64(temp1 - temp2)
            jm = np.int64(temp1 + temp2)

            ifp = jp >> hpix.factor_
            ifm = jm >> hpix.factor_

            if (ifp == ifm):
                face = 4 if ifp == 4 else ifp + 4
            elif (ifp < ifm):
                face = ifp
            else:
                face = ifm + 8

            x = jm & hpix.nsideminusone_
            y = hpix.nsideminusone_ - (jp & hpix.nsideminusone_)
        else:
            ntt = np.int64(tt)

            tp = tt - np.double(ntt)

            temp1 = hpix.dnside_ * rtz[i]

            jp = np.int64(tp * temp1)
            jm = np.int64((1.0 - tp) * temp1)

            if (jp >= hpix.nside_):
                jp = hpix.nsideminusone_

            if (jm >= hpix.nside_):
                jm = hpix.nsideminusone_

            if (z[i] >= 0):
                face = ntt
                x = hpix.nsideminusone_ - jm
                y = hpix.nsideminusone_ - jp
            else:
                face = ntt + 8
                x = jp
                y = jm

        sipf = hpix.xy2pix_(np.int64(x), np.int64(y))
        pix[i] = np.int64(sipf) + (face << (2 * hpix.factor_))


def zphi2ring(hpix, phi, region, z, rtz, pix):
    """
    Args:
        hpix (HealpixPixels):  The healpix projection object.
        phi (array, double) of size n
        region (array, int) of size n
        z (array, double) of size n
        rtz (array, double) of size n
        pix (array, int) of size n

    Returns:
        None (the results are put in pix)
    """
    # machine epsilon
    eps = np.finfo(np.float).eps

    # TODO that loop is a prime target for a vmap in jax
    n = pix.size
    for i in range(n):
        ph = phi[i]
        if (np.abs(ph) < eps):
            ph = 0.0

        tt = ph * TWOINVPI
        if (ph < 0.0):
            tt += 4.0

        if (np.abs(region[i]) == 1):
            temp1 = hpix.halfnside_ + hpix.dnside_ * tt
            temp2 = hpix.tqnside_ * z[i]

            jp = np.int64(temp1 - temp2)
            jm = np.int64(temp1 + temp2)

            ir = hpix.nsideplusone_ + jp - jm
            kshift = 1 - (ir & 1)

            ip = (jp + jm - hpix.nside_ + kshift + 1) >> 1
            ip = ip % hpix.fournside_

            pix[i] = hpix.ncap_ + ((ir - 1) * hpix.fournside_ + ip)
        else:
            tp = tt - np.floor(tt)

            temp1 = hpix.dnside_ * rtz[i]

            jp = np.int64(tp * temp1)
            jm = np.int64((1.0 - tp) * temp1)
            ir = jp + jm + 1
            ip = np.int64(tt * np.double(ir))
            longpart = np.int64(ip / (4 * ir))
            ip -= longpart

            pix[i] = (2 * ir * (ir - 1) + ip) if (region[i] >
                                                  0) else (hpix.npix_ - 2 * ir * (ir + 1) + ip)


def vec2zphi(vec):
    """
    Args:
        vec (array, double) of shape (n,3)

    Returns:
        (phi, region, z, rtz)
        phi (array, double) of size n
        region (array, int) of size n
        z (array, double) of size n
        rtz (array, double) of size n
    """
    z = vec[:, 2]
    za = np.abs(z)

    # region encodes BOTH the sign of Z and whether its
    # absolute value is greater than 2/3.
    itemps = np.where(z > 0.0, 1, -1)
    region = np.where(za <= 2./3., itemps, 2*itemps)

    work1 = 3.0 * (1.0 - za)
    rtz = np.sqrt(work1)

    work2 = vec[:, 0]
    work3 = vec[:, 1]
    phi = np.atan(work3, work2)

    return (phi, region, z, rtz)


def vec2nest(hpix, vec, pix):
    """
    Args:
        hpix (HealpixPixels):  The healpix projection object.
        vec (array, double) of shape (n,3)
        pix (array, int) of size n

    Returns:
        None, the result will be stored in pix
    """
    (phi, region, z, rtz) = vec2zphi(vec)
    zphi2nest(hpix, phi, region, z, rtz, pix)


def vec2ring(hpix, vec, pix):
    """
    Args:
        hpix (HealpixPixels):  The healpix projection object.
        vec (array, double) of shape (n,3)
        pix (array, int) of size n

    Returns:
        None, the result will be stored in pix
    """
    (phi, region, z, rtz) = vec2zphi(vec)
    zphi2ring(hpix, phi, region, z, rtz, pix)

# -------------------------------------------------------------------------------------------------
# C++


# see toast_math_healpix.cpp
"""
void toast::HealpixPixels::vec2zphi(int64_t n, double const * vec,
                                    double * phi, int * region, double * z,
                                    double * rtz) const 
{
    toast::AlignedVector <double> work1(n);
    toast::AlignedVector <double> work2(n);
    toast::AlignedVector <double> work3(n);

    for (int64_t i = 0; i < n; ++i) 
    {
        int64_t offset = 3 * i;

        // region encodes BOTH the sign of Z and whether its
        // absolute value is greater than 2/3.

        z[i] = vec[offset + 2];

        double za = ::fabs(z[i]);

        int itemp = (z[i] > 0.0) ? 1 : -1;

        region[i] = (za <= TWOTHIRDS) ? itemp : itemp + itemp;

        work1[i] = 3.0 * (1.0 - za);
        work3[i] = vec[offset + 1];
        work2[i] = vec[offset];
    }

    toast::vfast_sqrt(n, work1.data(), rtz);
    toast::vatan2(n, work3.data(), work2.data(), phi);
}

void toast::HealpixPixels::zphi2nest(int64_t n, double const * phi,
                                     int const * region, double const * z,
                                     double const * rtz, int64_t * pix) const 
{
    double eps = std::numeric_limits <float>::epsilon();
    
    for (int64_t i = 0; i < n; ++i) 
    {
        double ph = phi[i];
        if (fabs(ph) < eps) 
        {
            ph = 0.0;
        }
        double tt = (ph >= 0.0) ? ph * TWOINVPI : ph * TWOINVPI + 4.0;

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

        if (::abs(region[i]) == 1) 
        {
            temp1 = halfnside_ + dnside_ * tt;
            temp2 = tqnside_ * z[i];

            jp = static_cast <int64_t> (temp1 - temp2);
            jm = static_cast <int64_t> (temp1 + temp2);

            ifp = jp >> factor_;
            ifm = jm >> factor_;

            face;
            if (ifp == ifm) 
            {
                face = (ifp == 4) ? static_cast <int64_t> (4) : ifp + 4;
            } 
            else if (ifp < ifm) 
            {
                face = ifp;
            } 
            else 
            {
                face = ifm + 8;
            }

            x = jm & nsideminusone_;
            y = nsideminusone_ - (jp & nsideminusone_);
        } 
        else 
        {
            ntt = static_cast <int64_t> (tt);

            tp = tt - static_cast <double> (ntt);

            temp1 = dnside_ * rtz[i];

            jp = static_cast <int64_t> (tp * temp1);
            jm = static_cast <int64_t> ((1.0 - tp) * temp1);

            if (jp >= nside_) 
            {
                jp = nsideminusone_;
            }
            if (jm >= nside_) 
            {
                jm = nsideminusone_;
            }

            if (z[i] >= 0) 
            {
                face = ntt;
                x = nsideminusone_ - jm;
                y = nsideminusone_ - jp;
            } 
            else 
            {
                face = ntt + 8;
                x = jp;
                y = jm;
            }
        }

        uint64_t sipf = xy2pix_(static_cast <uint64_t> (x), static_cast <uint64_t> (y));
        pix[i] = static_cast <int64_t> (sipf) + (face << (2 * factor_));
    }
}

void toast::HealpixPixels::zphi2ring(int64_t n, double const * phi,
                                     int const * region, double const * z,
                                     double const * rtz, int64_t * pix) const {
    double eps = std::numeric_limits <float>::epsilon();
    
    for (int64_t i = 0; i < n; ++i) {
        double ph = phi[i];
        if (fabs(ph) < eps) {
            ph = 0.0;
        }
        double tt = (ph >= 0.0) ? ph * TWOINVPI : ph * TWOINVPI + 4.0;

        double tp;
        int64_t longpart;
        double temp1;
        double temp2;
        int64_t jp;
        int64_t jm;
        int64_t ip;
        int64_t ir;
        int64_t kshift;

        if (::abs(region[i]) == 1) {
            temp1 = halfnside_ + dnside_ * tt;
            temp2 = tqnside_ * z[i];

            jp = static_cast <int64_t> (temp1 - temp2);
            jm = static_cast <int64_t> (temp1 + temp2);

            ir = nsideplusone_ + jp - jm;
            kshift = 1 - (ir & 1);

            ip = (jp + jm - nside_ + kshift + 1) >> 1;
            ip = ip % fournside_;

            pix[i] = ncap_ + ((ir - 1) * fournside_ + ip);
        } else {
            tp = tt - floor(tt);

            temp1 = dnside_ * rtz[i];

            jp = static_cast <int64_t> (tp * temp1);
            jm = static_cast <int64_t> ((1.0 - tp) * temp1);
            ir = jp + jm + 1;
            ip = static_cast <int64_t> (tt * (double)ir);
            longpart = static_cast <int64_t> (ip / (4 * ir));
            ip -= longpart;

            pix[i] = (region[i] > 0) ? (2 * ir * (ir - 1) + ip)
                        : (npix_ - 2 * ir * (ir + 1) + ip);
        }
    }
}

void toast::HealpixPixels::vec2nest(int64_t n, double const * vec,
                                    int64_t * pix) const 
{
    toast::AlignedVector <double> z(n);
    toast::AlignedVector <double> rtz(n);
    toast::AlignedVector <double> phi(n);
    toast::AlignedVector <int> region(n);

    vec2zphi(n, vec, phi.data(), region.data(), z.data(), rtz.data());
    zphi2nest(n, phi.data(), region.data(), z.data(), rtz.data(), pix);
}

void toast::HealpixPixels::vec2ring(int64_t n, double const * vec,
                                    int64_t * pix) const {
    toast::AlignedVector <double> z(n);
    toast::AlignedVector <double> rtz(n);
    toast::AlignedVector <double> phi(n);
    toast::AlignedVector <int> region(n);

    vec2zphi(n, vec, phi.data(), region.data(), z.data(), rtz.data());
    zphi2ring(n, phi.data(), region.data(), z.data(), rtz.data(), pix);
}
"""
