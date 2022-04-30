
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// #if !defined(NO_ATM_CHECKS)
// # define NO_ATM_CHECKS
// #endif // if !defined(NO_ATM_CHECKS)

#include <toast/sys_utils.hpp>
#include <toast/sys_environment.hpp>
#include <toast/math_rng.hpp>
#include <toast/atm.hpp>

#include <sstream>
#include <iostream>
#include <fstream>
#include <cstring>
#include <random>
#include <functional>
#include <cmath>
#include <algorithm>


#ifdef HAVE_CHOLMOD


bool toast::atm_verbose() {
    // Helper function to check the log level once and return a static value.
    // This is useful to avoid expensive string operations inside deeply nested
    // functions.
    static bool called = false;
    static bool verbose = false;
    if (!called) {
        // First time we were called
        auto & env = toast::Environment::get();
        std::string logval = env.log_level();
        if (strncmp(logval.c_str(), "VERBOSE", 7) == 0) {
            verbose = true;
        }
        called = true;
    }
    return verbose;
}

toast::CholmodCommon & toast::CholmodCommon::get() {
    static toast::CholmodCommon instance;
    return instance;
}

toast::CholmodCommon::CholmodCommon() {
    // Initialize cholmod
    chcommon = &cholcommon;
    cholmod_start(chcommon);

    if (atm_verbose()) {
        // TK: What is the cost of this level?  Do we need another more verbose switch?
        chcommon->print = 3;          // Default verbosity
    } else {
        chcommon->print = 1;          // Minimal verbosity
    }
    chcommon->itype = CHOLMOD_INT;    // All integer arrays are int
    chcommon->dtype = CHOLMOD_DOUBLE; // All float arrays are double
    chcommon->final_ll = 1;           // The factorization is LL', not LDL'
}

toast::CholmodCommon::~CholmodCommon() {
    cholmod_finish(chcommon);
}

// This function is passing arguments by reference and is using precomputed values
// since it is called many times inside a loop.

bool toast::atm_sim_in_cone(
    double const & x,
    double const & y,
    double const & z,
    double const & t_in,
    double const & delta_t,
    double const & delta_az,
    double const & elmin,
    double const & elmax,
    double const & wx,
    double const & wy,
    double const & wz,
    double const & xstep,
    double const & ystep,
    double const & zstep,
    double const & maxdist,
    double const & cosel0,
    double const & sinel0
) {
    // Input coordinates are in the scan frame, rotate to horizontal frame

    double tstep = 1;

    for (double t = 0; t < delta_t; t += tstep) {
        if (t_in >= 0) {
            if (t != 0) {
                break;
            }
            t = t_in;
        }

        if ((t_in < 0) && (delta_t - t < tstep)) {
            t = delta_t;
        }

        double xtel_now = wx * t;
        double dx = x - xtel_now;

        // Is the point behind the telescope at this time?

        if (dx + xstep < 0) {
            if ((t_in >= 0) && atm_verbose()) {
                std::ostringstream o;
                o << "dx + xstep < 0: " << dx;
                auto & logger = toast::Logger::get();
                logger.warning(o.str().c_str());
            }
            continue;
        }

        // Check the rest of the spherical coordinates

        double ytel_now = wy * t;
        double dy = y - ytel_now;

        double ztel_now = wz * t;
        double dz = z - ztel_now;

        double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (r > maxdist * 1.01) {
            if ((t_in >= 0) && atm_verbose()) {
                std::ostringstream o;
                o << "r = " << r << " > maxdist = " << maxdist << " (dx = "
                  << dx << ", dy = " << dy << ", dz = " << dz << ")";
                auto & logger = toast::Logger::get();
                logger.warning(o.str().c_str());
            }
            continue;
        }

        if (dz > 0) {
            dz -= zstep;
        } else {
            dz += zstep;
        }

        if ((std::abs(dy) < 2 * ystep) && (std::abs(dz) < 2 * zstep)) {
            return true;
        }

        double dxx = dx * cosel0 - dz * sinel0;
        double dyy = dy;
        double dzz = dx * sinel0 + dz * cosel0;

        double el = std::asin(dzz / r);
        if ((el < elmin) || (el > elmax)) {
            if ((t_in >= 0) && atm_verbose()) {
                std::ostringstream o;
                o << "el outside cone: "
                  << el * 180 / M_PI << " not in "
                  << elmin * 180 / M_PI << " - "
                  << elmax * 180 / M_PI;
                auto & logger = toast::Logger::get();
                logger.warning(o.str().c_str());
            }
            continue;
        }

        dxx = (dx + xstep) * cosel0 - dz * sinel0;
        double az = std::atan2(dyy, dxx);
        if (std::abs(az) > 0.5 * delta_az) {
            if ((t_in >= 0) && atm_verbose()) {
                std::ostringstream o;
                o << "abs(az) > delta_az/2 "
                  << az * 180 / M_PI << " > "
                  << 0.5 * delta_az * 180 / M_PI;
                auto & logger = toast::Logger::get();
                logger.warning(o.str().c_str());
            }
            continue;
        }

        // Passed all the checks
        return true;
    }

    return false;
}

void toast::atm_sim_compress_flag_hits_rank(
    int64_t nn,
    uint8_t * hit,
    int ntask,
    int rank,
    int64_t nx,
    int64_t ny,
    int64_t nz,
    double xstart,
    double ystart,
    double zstart,
    double delta_t,
    double delta_az,
    double elmin,
    double elmax,
    double wx,
    double wy,
    double wz,
    double xstep,
    double ystep,
    double zstep,
    int64_t xstride,
    int64_t ystride,
    int64_t zstride,
    double maxdist,
    double cosel0,
    double sinel0
) {
    double t_fake = -1.0;
    for (int64_t ix = 0; ix < nx; ++ix) {
        if (ix % ntask != rank) {
            continue;
        }
        double x = xstart + ix * xstep;

        # pragma omp parallel for schedule(static, 4)
        for (int64_t iy = 0; iy < ny; ++iy) {
            double y = ystart + iy * ystep;

            for (int64_t iz = 0; iz < nz; ++iz) {
                double z = zstart + iz * zstep;
                if (toast::atm_sim_in_cone(
                        x, y, z, t_fake, delta_t, delta_az, elmin, elmax, wx, wy, wz,
                        xstep, ystep, zstep, maxdist, cosel0, sinel0)) {
                    hit[ix * xstride + iy * ystride + iz * zstride] = true;
                }
            }
        }
    }
    return;
}

void toast::atm_sim_compress_flag_extend_rank(
    uint8_t * hit,
    uint8_t * hit2,
    int ntask,
    int rank,
    int64_t nx,
    int64_t ny,
    int64_t nz,
    int64_t xstride,
    int64_t ystride,
    int64_t zstride
) {
    for (int64_t ix = 1; ix < nx - 1; ++ix) {
        if (ix % ntask != rank) {
            continue;
        }

        # pragma omp parallel for schedule(static, 4)
        for (int64_t iy = 1; iy < ny; ++iy) {
            for (int64_t iz = 1; iz < nz; ++iz) {
                int64_t offset = ix * xstride + iy * ystride + iz * zstride;

                if (hit2[offset]) {
                    // Flag this element but also its neighbours to facilitate
                    // interpolation

                    for (int64_t xmul = -2; xmul < 4; ++xmul) {
                        if ((ix + xmul < 0) || (ix + xmul > nx - 1)) continue;

                        for (int64_t ymul = -2; ymul < 4; ++ymul) {
                            if ((iy + ymul < 0) ||
                                (iy + ymul > ny - 1)) continue;

                            for (int64_t zmul = -2; zmul < 4; ++zmul) {
                                if ((iz + zmul < 0) ||
                                    (iz + zmul > nz - 1)) continue;
                                hit[offset + xmul * xstride
                                    + ymul * ystride + zmul * zstride] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    return;
}

#endif // ifdef HAVE_CHOLMOD
