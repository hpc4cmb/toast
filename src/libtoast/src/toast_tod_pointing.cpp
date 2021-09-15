
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/math_qarray.hpp>
#include <toast/math_sf.hpp>
#include <toast/tod_pointing.hpp>

#include <sstream>
#include <iostream>


void toast::pointing_matrix_healpix(toast::HealpixPixels const & hpix,
                                    bool nest, double eps, double cal,
                                    std::string const & mode, size_t n,
                                    double const * pdata,
                                    double const * hwpang,
                                    uint8_t const * flags,
                                    int64_t * pixels, double * weights) {
    double xaxis[3] = {1.0, 0.0, 0.0};
    double zaxis[3] = {0.0, 0.0, 1.0};
    double nullquat[4] = {0.0, 0.0, 0.0, 1.0};

    double eta = (1.0 - eps) / (1.0 + eps);

    toast::AlignedVector <double> dir(3 * n);
    toast::AlignedVector <double> pin(4 * n);

    if (flags == NULL) {
        std::copy(pdata, pdata + (4 * n), pin.begin());
    } else {
        size_t off;
        for (size_t i = 0; i < n; ++i) {
            off = 4 * i;
            if (flags[i] == 0) {
                pin[off] = pdata[off];
                pin[off + 1] = pdata[off + 1];
                pin[off + 2] = pdata[off + 2];
                pin[off + 3] = pdata[off + 3];
            } else {
                pin[off] = nullquat[0];
                pin[off + 1] = nullquat[1];
                pin[off + 2] = nullquat[2];
                pin[off + 3] = nullquat[3];
            }
        }
    }

    toast::qa_rotate_many_one(n, pin.data(), zaxis, dir.data());

    if (nest) {
        hpix.vec2nest(n, dir.data(), pixels);
    } else {
        hpix.vec2ring(n, dir.data(), pixels);
    }

    if (flags != NULL) {
        for (size_t i = 0; i < n; ++i) {
            pixels[i] = (flags[i] == 0) ? pixels[i] : -1;
        }
    }

    if (mode == "I") {
        for (size_t i = 0; i < n; ++i) {
            weights[i] = cal;
        }
    } else if (mode == "IQU") {
        toast::AlignedVector <double> orient(3 * n);
        toast::AlignedVector <double> buf1(n);
        toast::AlignedVector <double> buf2(n);

        toast::qa_rotate_many_one(n, pin.data(), xaxis, orient.data());

        double * bx = buf1.data();
        double * by = buf2.data();

        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            size_t off = 3 * i;
            by[i] = orient[off + 0] * dir[off + 1] - orient[off + 1] *
                    dir[off + 0];
            bx[i] = orient[off + 0] * (-dir[off + 2] * dir[off + 0]) +
                    orient[off + 1] * (-dir[off + 2] * dir[off + 1]) +
                    orient[off + 2] * (dir[off + 0] * dir[off + 0] +
                                       dir[off + 1] * dir[off + 1]);
        }

        toast::AlignedVector <double> detang(n);

        // FIXME:  Switch back to fast version after unit tests improved.
        toast::vatan2(n, by, bx, detang.data());

        if (hwpang == NULL) {
            for (size_t i = 0; i < n; ++i) {
                detang[i] *= 2.0;
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                detang[i] += 2.0 * hwpang[i];
                detang[i] *= 2.0;
            }
        }

        double * sinout = buf1.data();
        double * cosout = buf2.data();

        // FIXME:  Switch back to fast version after unit tests pass
        toast::vsincos(n, detang.data(), sinout, cosout);

        for (size_t i = 0; i < n; ++i) {
            size_t off = 3 * i;
            weights[off + 0] = cal;
            weights[off + 1] = cosout[i] * eta * cal;
            weights[off + 2] = sinout[i] * eta * cal;
        }
    } else {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "unknown healpix pointing matrix mode \"" << mode << "\"";
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }

    return;
}

void toast::healpix_pixels(toast::HealpixPixels const & hpix, bool nest,
                           size_t n, double const * pdata,
                           uint8_t const * flags, int64_t * pixels) {
    double zaxis[3] = {0.0, 0.0, 1.0};
    double nullquat[4] = {0.0, 0.0, 0.0, 1.0};

    toast::AlignedVector <double> dir(3 * n);
    toast::AlignedVector <double> pin(4 * n);

    if (flags == NULL) {
        std::copy(pdata, pdata + (4 * n), pin.begin());
    } else {
        size_t off;
        for (size_t i = 0; i < n; ++i) {
            off = 4 * i;
            if (flags[i] == 0) {
                pin[off] = pdata[off];
                pin[off + 1] = pdata[off + 1];
                pin[off + 2] = pdata[off + 2];
                pin[off + 3] = pdata[off + 3];
            } else {
                pin[off] = nullquat[0];
                pin[off + 1] = nullquat[1];
                pin[off + 2] = nullquat[2];
                pin[off + 3] = nullquat[3];
            }
        }
    }

    toast::qa_rotate_many_one(n, pin.data(), zaxis, dir.data());

    if (nest) {
        hpix.vec2nest(n, dir.data(), pixels);
    } else {
        hpix.vec2ring(n, dir.data(), pixels);
    }

    if (flags != NULL) {
        for (size_t i = 0; i < n; ++i) {
            pixels[i] = (flags[i] == 0) ? pixels[i] : -1;
        }
    }

    return;
}

void toast::stokes_weights(double eps, double cal, std::string const & mode,
                           size_t n, double const * pdata,
                           double const * hwpang,  uint8_t const * flags,
                           double * weights) {
    double xaxis[3] = {1.0, 0.0, 0.0};
    double zaxis[3] = {0.0, 0.0, 1.0};
    double nullquat[4] = {0.0, 0.0, 0.0, 1.0};

    double eta = (1.0 - eps) / (1.0 + eps);

    toast::AlignedVector <double> dir(3 * n);
    toast::AlignedVector <double> pin(4 * n);

    if (flags == NULL) {
        std::copy(pdata, pdata + (4 * n), pin.begin());
    } else {
        size_t off;
        for (size_t i = 0; i < n; ++i) {
            off = 4 * i;
            if (flags[i] == 0) {
                pin[off] = pdata[off];
                pin[off + 1] = pdata[off + 1];
                pin[off + 2] = pdata[off + 2];
                pin[off + 3] = pdata[off + 3];
            } else {
                pin[off] = nullquat[0];
                pin[off + 1] = nullquat[1];
                pin[off + 2] = nullquat[2];
                pin[off + 3] = nullquat[3];
            }
        }
    }

    toast::qa_rotate_many_one(n, pin.data(), zaxis, dir.data());

    if (mode == "I") {
        for (size_t i = 0; i < n; ++i) {
            weights[i] = cal;
        }
    } else if (mode == "IQU") {
        toast::AlignedVector <double> orient(3 * n);
        toast::AlignedVector <double> buf1(n);
        toast::AlignedVector <double> buf2(n);

        toast::qa_rotate_many_one(n, pin.data(), xaxis, orient.data());

        double * bx = buf1.data();
        double * by = buf2.data();

        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            size_t off = 3 * i;
            by[i] = orient[off + 0] * dir[off + 1] - orient[off + 1] *
                    dir[off + 0];
            bx[i] = orient[off + 0] * (-dir[off + 2] * dir[off + 0]) +
                    orient[off + 1] * (-dir[off + 2] * dir[off + 1]) +
                    orient[off + 2] * (dir[off + 0] * dir[off + 0] +
                                       dir[off + 1] * dir[off + 1]);
        }

        toast::AlignedVector <double> detang(n);

        // FIXME:  Switch back to fast version after unit tests improved.
        toast::vatan2(n, by, bx, detang.data());

        if (hwpang == NULL) {
            for (size_t i = 0; i < n; ++i) {
                detang[i] *= 2.0;
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                detang[i] += 2.0 * hwpang[i];
                detang[i] *= 2.0;
            }
        }

        double * sinout = buf1.data();
        double * cosout = buf2.data();

        // FIXME:  Switch back to fast version after unit tests pass
        toast::vsincos(n, detang.data(), sinout, cosout);

        for (size_t i = 0; i < n; ++i) {
            size_t off = 3 * i;
            weights[off + 0] = cal;
            weights[off + 1] = cosout[i] * eta * cal;
            weights[off + 2] = sinout[i] * eta * cal;
        }
    } else {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "unknown stokes weights mode \"" << mode << "\"";
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }

    return;
}
