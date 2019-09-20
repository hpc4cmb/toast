
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/math_sf.hpp>
#include <toast/math_healpix.hpp>

#include <cmath>


const int64_t toast::HealpixPixels::jr_[] =
{2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};

const int64_t toast::HealpixPixels::jp_[] =
{1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};


void toast::healpix_ang2vec(int64_t n, double const * theta,
                            double const * phi, double * vec) {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    if (toast::is_aligned(theta) && toast::is_aligned(phi) &&
        toast::is_aligned(vec)) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            int64_t offset = 3 * i;
            double sintheta = ::sin(theta[i]);
            vec[offset] = sintheta * ::cos(phi[i]);
            vec[offset + 1] = sintheta * ::sin(phi[i]);
            vec[offset + 2] = ::cos(theta[i]);
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            int64_t offset = 3 * i;
            double sintheta = ::sin(theta[i]);
            vec[offset] = sintheta * ::cos(phi[i]);
            vec[offset + 1] = sintheta * ::sin(phi[i]);
            vec[offset + 2] = ::cos(theta[i]);
        }
    }

    return;
}

void toast::healpix_vec2ang(int64_t n, double const * vec, double * theta,
                            double * phi) {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    if (toast::is_aligned(theta) && toast::is_aligned(phi) &&
        toast::is_aligned(vec)) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            int64_t offset = 3 * i;
            double norm = 1.0 / ::sqrt(vec[offset] * vec[offset]
                                       + vec[offset + 1] * vec[offset + 1]
                                       + vec[offset + 2] * vec[offset + 2]);
            theta[i] = ::acos(vec[offset + 2] * norm);
            double phitemp = ::atan2(vec[offset + 1], vec[offset]);
            phi[i] = (phitemp < 0) ? phitemp + toast::TWOPI : phitemp;
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            int64_t offset = 3 * i;
            double norm = 1.0 / ::sqrt(vec[offset] * vec[offset]
                                       + vec[offset + 1] * vec[offset + 1]
                                       + vec[offset + 2] * vec[offset + 2]);
            theta[i] = ::acos(vec[offset + 2] * norm);
            double phitemp = ::atan2(vec[offset + 1], vec[offset]);
            phi[i] = (phitemp < 0) ? phitemp + toast::TWOPI : phitemp;
        }
    }

    return;
}

void toast::healpix_vecs2angpa(int64_t n, double const * vec, double * theta,
                               double * phi, double * pa) {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    if (toast::is_aligned(vec) && toast::is_aligned(theta) &&
        toast::is_aligned(phi) && toast::is_aligned(pa)) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            int64_t offset = 6 * i;
            double dx = vec[offset];
            double dy = vec[offset + 1];
            double dz = vec[offset + 2];
            double ox = vec[offset + 3];
            double oy = vec[offset + 4];
            double oz = vec[offset + 5];

            double xysq = dx * dx + dy * dy;

            double ypa = ox * dy - oy * dx;
            double xpa = -(ox * dz * dx) - (oy * dz * dy) + (oz * xysq);

            double xy = ::sqrt(xysq);

            theta[i] = ::atan2(xy, dz);
            double phitemp = ::atan2(dy, dx);
            pa[i] = ::atan2(ypa, xpa);

            phi[i] = (phitemp < 0) ? phitemp + toast::TWOPI : phitemp;
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            int64_t offset = 6 * i;
            double dx = vec[offset];
            double dy = vec[offset + 1];
            double dz = vec[offset + 2];
            double ox = vec[offset + 3];
            double oy = vec[offset + 4];
            double oz = vec[offset + 5];

            double xysq = dx * dx + dy * dy;

            double ypa = ox * dy - oy * dx;
            double xpa = -(ox * dz * dx) - (oy * dz * dy) + (oz * xysq);

            double xy = ::sqrt(xysq);

            theta[i] = ::atan2(xy, dz);
            double phitemp = ::atan2(dy, dx);
            pa[i] = ::atan2(ypa, xpa);

            phi[i] = (phitemp < 0) ? phitemp + toast::TWOPI : phitemp;
        }
    }

    return;
}

void toast::HealpixPixels::init() {
    nside_ = 0;
    ncap_ = 0;
    npix_ = 0;
    dnside_ = 0.0;
    fournside_ = 0;
    twonside_ = 0;
    nsideplusone_ = 0;
    halfnside_ = 0.0;
    tqnside_ = 0.0;
    factor_ = 0;
    nsideminusone_ = 0;
}

toast::HealpixPixels::HealpixPixels() {
    init();
    reset(1);
}

toast::HealpixPixels::HealpixPixels(int64_t nside) {
    init();
    reset(nside);
}

void toast::HealpixPixels::reset(int64_t nside) {
    if (nside <= 0) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("cannot reset healpix pixels with NSIDE <= 0");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    // check for valid nside value

    uint64_t temp = static_cast <uint64_t> (nside);
    if (((~temp) & (temp - 1)) != (temp - 1)) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("invalid NSIDE value- must be a multiple of 2");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    nside_ = nside;

    for (uint64_t m = 0; m < 0x100; ++m) {
        utab_[m] = (m & 0x1) | ((m & 0x2) << 1) | ((m & 0x4) << 2) |
                   ((m & 0x8) << 3) | ((m & 0x10) << 4) | ((m & 0x20) << 5) |
                   ((m & 0x40) << 6) | ((m & 0x80) << 7);

        ctab_[m] = (m & 0x1) | ((m & 0x2) << 7) | ((m & 0x4) >> 1) |
                   ((m & 0x8) << 6) | ((m & 0x10) >> 2) | ((m & 0x20) << 5) |
                   ((m & 0x40) >> 3) | ((m & 0x80) << 4);
    }

    ncap_ = 2 * (nside * nside - nside);

    npix_ = 12 * nside * nside;

    dnside_ = static_cast <double> (nside);

    twonside_ = 2 * nside;

    fournside_ = 4 * nside;

    nsideplusone_ = nside + 1;

    halfnside_ = 0.5 * (dnside_);

    tqnside_ = 0.75 * (dnside_);

    factor_ = 0;

    nsideminusone_ = nside - 1;

    while (nside != (1ll << factor_)) {
        ++factor_;
    }

    return;
}

void toast::HealpixPixels::vec2zphi(int64_t n, double const * vec,
                                    double * phi, int * region, double * z,
                                    double * rtz) const {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    toast::AlignedVector <double> work1(n);
    toast::AlignedVector <double> work2(n);
    toast::AlignedVector <double> work3(n);

    if (toast::is_aligned(vec) && toast::is_aligned(phi) &&
        toast::is_aligned(region) && toast::is_aligned(z)
        && toast::is_aligned(rtz)) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
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
    } else {
        for (int64_t i = 0; i < n; ++i) {
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
    }

    toast::vfast_sqrt(n, work1.data(), rtz);

    // FIXME:  revert to fast version after unit tests in place
    // toast::vfast_atan2(n, work3.data(), work2.data(), phi);
    toast::vatan2(n, work3.data(), work2.data(), phi);

    return;
}

void toast::HealpixPixels::theta2z(int64_t n, double const * theta,
                                   int * region, double * z,
                                   double * rtz) const {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    toast::AlignedVector <double> work1(n);

    // FIXME:  revert to fast version once unit tests pass
    // toast::vcos(static_cast <int> (n), theta, z);
    toast::vfast_cos(static_cast <int> (n), theta, z);

    if (toast::is_aligned(theta) && toast::is_aligned(region)
        && toast::is_aligned(z) && toast::is_aligned(rtz)) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            // region encodes BOTH the sign of Z and whether its
            // absolute value is greater than 2/3.

            double za = ::fabs(z[i]);

            int itemp = (z[i] > 0.0) ? 1 : -1;

            region[i] = (za <= TWOTHIRDS) ? itemp : itemp + itemp;

            work1[i] = 3.0 * (1.0 - za);
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            // region encodes BOTH the sign of Z and whether its
            // absolute value is greater than 2/3.

            double za = ::fabs(z[i]);

            int itemp = (z[i] > 0.0) ? 1 : -1;

            region[i] = (za <= TWOTHIRDS) ? itemp : itemp + itemp;

            work1[i] = 3.0 * (1.0 - za);
        }
    }

    toast::vfast_sqrt(n, work1.data(), rtz);

    return;
}

void toast::HealpixPixels::zphi2nest(int64_t n, double const * phi,
                                     int const * region, double const * z,
                                     double const * rtz, int64_t * pix) const {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    if (toast::is_aligned(phi) && toast::is_aligned(pix) &&
        toast::is_aligned(region) && toast::is_aligned(z)
        && toast::is_aligned(rtz)) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            double tt =
                (phi[i] >= 0.0) ? phi[i] * TWOINVPI : phi[i] * TWOINVPI + 4.0;

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

            if (::abs(region[i]) == 1) {
                temp1 = halfnside_ + dnside_ * tt;
                temp2 = tqnside_ * z[i];

                jp = static_cast <int64_t> (temp1 - temp2);
                jm = static_cast <int64_t> (temp1 + temp2);

                ifp = jp >> factor_;
                ifm = jm >> factor_;

                face;
                if (ifp == ifm) {
                    face = (ifp == 4) ? static_cast <int64_t> (4) : ifp + 4;
                } else if (ifp < ifm) {
                    face = ifp;
                } else {
                    face = ifm + 8;
                }

                x = jm & nsideminusone_;
                y = nsideminusone_ - (jp & nsideminusone_);
            } else {
                ntt = static_cast <int64_t> (tt);

                tp = tt - static_cast <double> (ntt);

                temp1 = dnside_ * rtz[i];

                jp = static_cast <int64_t> (tp * temp1);
                jm = static_cast <int64_t> ((1.0 - tp) * temp1);

                if (jp >= nside_) {
                    jp = nsideminusone_;
                }
                if (jm >= nside_) {
                    jm = nsideminusone_;
                }

                if (z[i] >= 0) {
                    face = ntt;
                    x = nsideminusone_ - jm;
                    y = nsideminusone_ - jp;
                } else {
                    face = ntt + 8;
                    x = jp;
                    y = jm;
                }
            }

            uint64_t sipf = xy2pix_(static_cast <uint64_t> (x),
                                    static_cast <uint64_t> (y));

            pix[i] = static_cast <int64_t> (sipf) + (face << (2 * factor_));
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            double tt =
                (phi[i] >= 0.0) ? phi[i] * TWOINVPI : phi[i] * TWOINVPI + 4.0;

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

            if (::abs(region[i]) == 1) {
                temp1 = halfnside_ + dnside_ * tt;
                temp2 = tqnside_ * z[i];

                jp = static_cast <int64_t> (temp1 - temp2);
                jm = static_cast <int64_t> (temp1 + temp2);

                ifp = jp >> factor_;
                ifm = jm >> factor_;

                face;
                if (ifp == ifm) {
                    face = (ifp == 4) ? static_cast <int64_t> (4) : ifp + 4;
                } else if (ifp < ifm) {
                    face = ifp;
                } else {
                    face = ifm + 8;
                }

                x = jm & nsideminusone_;
                y = nsideminusone_ - (jp & nsideminusone_);
            } else {
                ntt = static_cast <int64_t> (tt);

                tp = tt - static_cast <double> (ntt);

                temp1 = dnside_ * rtz[i];

                jp = static_cast <int64_t> (tp * temp1);
                jm = static_cast <int64_t> ((1.0 - tp) * temp1);

                if (jp >= nside_) {
                    jp = nsideminusone_;
                }
                if (jm >= nside_) {
                    jm = nsideminusone_;
                }

                if (z[i] >= 0) {
                    face = ntt;
                    x = nsideminusone_ - jm;
                    y = nsideminusone_ - jp;
                } else {
                    face = ntt + 8;
                    x = jp;
                    y = jm;
                }
            }

            uint64_t sipf = xy2pix_(static_cast <uint64_t> (x),
                                    static_cast <uint64_t> (y));

            pix[i] = static_cast <int64_t> (sipf) + (face << (2 * factor_));
        }
    }

    return;
}

void toast::HealpixPixels::zphi2ring(int64_t n, double const * phi,
                                     int const * region, double const * z,
                                     double const * rtz, int64_t * pix) const {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    if (toast::is_aligned(phi) && toast::is_aligned(pix) &&
        toast::is_aligned(region) && toast::is_aligned(z)
        && toast::is_aligned(rtz)) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            double tt =
                (phi[i] >= 0.0) ? phi[i] * TWOINVPI : phi[i] * TWOINVPI + 4.0;

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
    } else {
        for (int64_t i = 0; i < n; ++i) {
            double tt =
                (phi[i] >= 0.0) ? phi[i] * TWOINVPI : phi[i] * TWOINVPI + 4.0;

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

    return;
}

void toast::HealpixPixels::ang2nest(int64_t n, double const * theta,
                                    double const * phi, int64_t * pix) const {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    toast::AlignedVector <double> z(n);
    toast::AlignedVector <double> rtz(n);
    toast::AlignedVector <int> region(n);

    theta2z(n, theta, region.data(), z.data(), rtz.data());

    zphi2nest(n, phi, region.data(), z.data(), rtz.data(), pix);

    return;
}

void toast::HealpixPixels::ang2ring(int64_t n, double const * theta,
                                    double const * phi, int64_t * pix) const {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    toast::AlignedVector <double> z(n);
    toast::AlignedVector <double> rtz(n);
    toast::AlignedVector <int> region(n);

    theta2z(n, theta, region.data(), z.data(), rtz.data());

    zphi2ring(n, phi, region.data(), z.data(), rtz.data(), pix);

    return;
}

void toast::HealpixPixels::vec2nest(int64_t n, double const * vec,
                                    int64_t * pix) const {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    toast::AlignedVector <double> z(n);
    toast::AlignedVector <double> rtz(n);
    toast::AlignedVector <double> phi(n);
    toast::AlignedVector <int> region(n);

    vec2zphi(n, vec, phi.data(), region.data(), z.data(), rtz.data());

    zphi2nest(n, phi.data(), region.data(), z.data(), rtz.data(), pix);

    return;
}

void toast::HealpixPixels::vec2ring(int64_t n, double const * vec,
                                    int64_t * pix) const {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    toast::AlignedVector <double> z(n);
    toast::AlignedVector <double> rtz(n);
    toast::AlignedVector <double> phi(n);
    toast::AlignedVector <int> region(n);

    vec2zphi(n, vec, phi.data(), region.data(), z.data(), rtz.data());

    zphi2ring(n, phi.data(), region.data(), z.data(), rtz.data(), pix);

    return;
}

void toast::HealpixPixels::ring2nest(int64_t n, int64_t const * ringpix,
                                     int64_t * nestpix) const {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    if (toast::is_aligned(ringpix) && toast::is_aligned(nestpix)) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            int64_t fc;
            uint64_t x, y;
            int64_t ix, iy;
            int64_t nr;
            int64_t kshift;
            int64_t iring;
            int64_t iphi;
            int64_t tmp;
            int64_t ip;
            int64_t ire, irm;
            int64_t ifm, ifp;
            int64_t irt, ipt;
            if (ringpix[i] < ncap_) {
                iring = static_cast <int64_t> (
                    0.5 * (1.0 +
                           ::sqrt(static_cast <double> (1 + 2 * ringpix[i]))));
                iphi  = (ringpix[i] + 1) - 2 * iring * (iring - 1);
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
            } else if (ringpix[i] < (npix_ - ncap_)) {
                ip = ringpix[i] - ncap_;
                iring = (ip >> (factor_ + 2)) + nside_;
                iphi = (ip & (fournside_ - 1)) + 1;
                kshift = (iring + nside_) & 1;
                nr = nside_;
                ire = iring - nside_ + 1;
                irm = twonside_ + 2 - ire;
                ifm = (iphi - (ire / 2) + nside_ - 1) >> factor_;
                ifp = (iphi - (irm / 2) + nside_ - 1) >> factor_;
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
                ip = npix_ - ringpix[i];
                iring = static_cast <int64_t> (
                    0.5 * (1.0 + ::sqrt(static_cast <double> (2 * ip - 1))));
                iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
                kshift = 0;
                nr = iring;
                iring = fournside_ - iring;
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

            irt = iring - jr_[fc] * nside_ + 1;
            ipt = 2 * iphi - jp_[fc] * nr - kshift - 1;
            if (ipt >= twonside_) {
                ipt -= 8 * nside_;
            }

            ix = (ipt - irt) >> 1;
            iy = (-(ipt + irt)) >> 1;
            x = static_cast <uint64_t> (ix);
            y = static_cast <uint64_t> (iy);

            nestpix[i] = xy2pix_(x, y);
            nestpix[i] += (fc << (2 * factor_));
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            int64_t fc;
            uint64_t x, y;
            int64_t ix, iy;
            int64_t nr;
            int64_t kshift;
            int64_t iring;
            int64_t iphi;
            int64_t tmp;
            int64_t ip;
            int64_t ire, irm;
            int64_t ifm, ifp;
            int64_t irt, ipt;
            if (ringpix[i] < ncap_) {
                iring = static_cast <int64_t> (
                    0.5 * (1.0 +
                           ::sqrt(static_cast <double> (1 + 2 * ringpix[i]))));
                iphi  = (ringpix[i] + 1) - 2 * iring * (iring - 1);
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
            } else if (ringpix[i] < (npix_ - ncap_)) {
                ip = ringpix[i] - ncap_;
                iring = (ip >> (factor_ + 2)) + nside_;
                iphi = (ip & (fournside_ - 1)) + 1;
                kshift = (iring + nside_) & 1;
                nr = nside_;
                ire = iring - nside_ + 1;
                irm = twonside_ + 2 - ire;
                ifm = (iphi - (ire / 2) + nside_ - 1) >> factor_;
                ifp = (iphi - (irm / 2) + nside_ - 1) >> factor_;
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
                ip = npix_ - ringpix[i];
                iring = static_cast <int64_t> (
                    0.5 * (1.0 + ::sqrt(static_cast <double> (2 * ip - 1))));
                iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
                kshift = 0;
                nr = iring;
                iring = fournside_ - iring;
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

            irt = iring - jr_[fc] * nside_ + 1;
            ipt = 2 * iphi - jp_[fc] * nr - kshift - 1;
            if (ipt >= twonside_) {
                ipt -= 8 * nside_;
            }

            ix = (ipt - irt) >> 1;
            iy = (-(ipt + irt)) >> 1;
            x = static_cast <uint64_t> (ix);
            y = static_cast <uint64_t> (iy);

            nestpix[i] = xy2pix_(x, y);
            nestpix[i] += (fc << (2 * factor_));
        }
    }

    return;
}

void toast::HealpixPixels::nest2ring(int64_t n, int64_t const * nestpix,
                                     int64_t * ringpix) const {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }
    if (toast::is_aligned(ringpix) && toast::is_aligned(nestpix)) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            int64_t fc;
            uint64_t x, y;
            int64_t ix, iy;
            int64_t jr;
            int64_t jp;
            int64_t nr;
            int64_t kshift;
            int64_t n_before;
            fc = nestpix[i] >> (2 * factor_);
            pix2xy_(nestpix[i] & (nside_ * nside_ - 1), x, y);
            ix = static_cast <int64_t> (x);
            iy = static_cast <int64_t> (y);

            jr = (jr_[fc] * nside_) - ix - iy - 1;

            if (jr < nside_) {
                nr = jr;
                n_before = 2 * nr * (nr - 1);
                kshift = 0;
            } else if (jr > (3 * nside_)) {
                nr = fournside_ - jr;
                n_before = npix_ - 2 * (nr + 1) * nr;
                kshift = 0;
            } else {
                nr = nside_;
                n_before = ncap_ + (jr - nside_) * fournside_;
                kshift = (jr - nside_) & 1;
            }

            jp = (jp_[fc] * nr + ix - iy + 1 + kshift) / 2;

            if (jp > fournside_) {
                jp -= fournside_;
            } else {
                if (jp < 1) {
                    jp += fournside_;
                }
            }

            ringpix[i] = n_before + jp - 1;
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            int64_t fc;
            uint64_t x, y;
            int64_t ix, iy;
            int64_t jr;
            int64_t jp;
            int64_t nr;
            int64_t kshift;
            int64_t n_before;
            fc = nestpix[i] >> (2 * factor_);
            pix2xy_(nestpix[i] & (nside_ * nside_ - 1), x, y);
            ix = static_cast <int64_t> (x);
            iy = static_cast <int64_t> (y);

            jr = (jr_[fc] * nside_) - ix - iy - 1;

            if (jr < nside_) {
                nr = jr;
                n_before = 2 * nr * (nr - 1);
                kshift = 0;
            } else if (jr > (3 * nside_)) {
                nr = fournside_ - jr;
                n_before = npix_ - 2 * (nr + 1) * nr;
                kshift = 0;
            } else {
                nr = nside_;
                n_before = ncap_ + (jr - nside_) * fournside_;
                kshift = (jr - nside_) & 1;
            }

            jp = (jp_[fc] * nr + ix - iy + 1 + kshift) / 2;

            if (jp > fournside_) {
                jp -= fournside_;
            } else {
                if (jp < 1) {
                    jp += fournside_;
                }
            }

            ringpix[i] = n_before + jp - 1;
        }
    }

    return;
}

void toast::HealpixPixels::degrade_ring(int factor, int64_t n,
                                        int64_t const * inpix,
                                        int64_t * outpix) const {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    toast::AlignedVector <int64_t> temp_nest(n);
    toast::AlignedVector <int64_t> temp(n);

    ring2nest(n, inpix, temp_nest.data());

    degrade_nest(factor, n, temp_nest.data(), temp.data());

    nest2ring(n, temp.data(), outpix);

    return;
}

void toast::HealpixPixels::degrade_nest(int factor, int64_t n,
                                        int64_t const * inpix,
                                        int64_t * outpix) const {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    int64_t shift = 2 * factor;

    if (toast::is_aligned(inpix) && toast::is_aligned(outpix)) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            outpix[i] = inpix[i] >> shift;
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            outpix[i] = inpix[i] >> shift;
        }
    }

    return;
}

void toast::HealpixPixels::upgrade_ring(int factor, int64_t n,
                                        int64_t const * inpix,
                                        int64_t * outpix) const {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    toast::AlignedVector <int64_t> temp_nest(n);
    toast::AlignedVector <int64_t> temp(n);

    ring2nest(n, inpix, temp_nest.data());

    upgrade_nest(factor, n, temp_nest.data(), temp.data());

    nest2ring(n, temp.data(), outpix);

    return;
}

void toast::HealpixPixels::upgrade_nest(int factor, int64_t n,
                                        int64_t const * inpix,
                                        int64_t * outpix) const {
    if (n > std::numeric_limits <int>::max()) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    int64_t shift = 2 * factor;

    if (toast::is_aligned(inpix) && toast::is_aligned(outpix)) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            outpix[i] = inpix[i] << shift;
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            outpix[i] = inpix[i] << shift;
        }
    }

    return;
}
