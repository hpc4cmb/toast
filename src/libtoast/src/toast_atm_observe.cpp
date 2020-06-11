
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


double toast::atm_sim_interp(
    double const & x,
    double const & y,
    double const & z,
    std::vector <int64_t> & last_ind,
    std::vector <double> & last_nodes,
    double const & xstart,
    double const & ystart,
    double const & zstart,
    int64_t const & nn,
    int64_t const & nx,
    int64_t const & ny,
    int64_t const & nz,
    int64_t const & xstride,
    int64_t const & ystride,
    int64_t const & zstride,
    double const & xstep,
    double const & ystep,
    double const & zstep,
    double const & xstepinv,
    double const & ystepinv,
    double const & zstepinv,
    double const & t_in,
    double const & delta_t,
    double const & delta_az,
    double const & elmin,
    double const & elmax,
    double const & wx,
    double const & wy,
    double const & wz,
    double const & maxdist,
    double const & cosel0,
    double const & sinel0,
    int64_t const & nelem,
    int64_t const * compressed_index,
    int64_t const * full_index,
    double const * realization
    ) {
    // Trilinear interpolation.  This function is called for every sample, so we
    // pass all arguments by reference / pointer.

    int64_t ix = (x - xstart) * xstepinv;
    int64_t iy = (y - ystart) * ystepinv;
    int64_t iz = (z - zstart) * zstepinv;

    double dx = (x - (xstart + (double)ix * xstep)) * xstepinv;
    double dy = (y - (ystart + (double)iy * ystep)) * ystepinv;
    double dz = (z - (zstart + (double)iz * zstep)) * zstepinv;

# ifndef NO_ATM_CHECKS
    if ((dx < 0) || (dx > 1) || (dy < 0) || (dy > 1) || (dz < 0) || (dz > 1)) {
        std::ostringstream o;
        o.precision(16);
        o << "atm_sim_interp : bad fractional step: " << std::endl
          << "x = " << x << std::endl
          << "y = " << y << std::endl
          << "z = " << z << std::endl
          << "dx = " << dx << std::endl
          << "dy = " << dy << std::endl
          << "dz = " << dz << std::endl;
        auto & logger = toast::Logger::get();
        logger.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
# endif // ifndef NO_ATM_CHECKS

    double c000, c001, c010, c011, c100, c101, c110, c111;

    if ((ix != last_ind[0]) || (iy != last_ind[1]) || (iz != last_ind[2])) {
# ifndef NO_ATM_CHECKS
        if ((ix < 0) || (ix > nx - 2) || (iy < 0) || (iy > ny - 2)
            || (iz < 0) || (iz > nz - 2)) {
            std::ostringstream o;
            o.precision(16);
            o << "atm_sim_interp : full index out of bounds at"
              << std::endl << "("
              << x << ", " << y << ", " << z << ") = ("
              << ix << "/" << nx << ", "
              << iy << "/" << ny << ", "
              << iz << "/" << nz << ")";
            auto & logger = toast::Logger::get();
            logger.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
# endif // ifndef NO_ATM_CHECKS

        int64_t offset = ix * xstride + iy * ystride + iz * zstride;

        int64_t ifull000 = offset;
        int64_t ifull001 = offset + zstride;
        int64_t ifull010 = offset + ystride;
        int64_t ifull011 = ifull010 + zstride;
        int64_t ifull100 = offset + xstride;
        int64_t ifull101 = ifull100 + zstride;
        int64_t ifull110 = ifull100 + ystride;
        int64_t ifull111 = ifull110 + zstride;

# ifndef NO_ATM_CHECKS
        int64_t ifullmax = nn - 1;
        if (
            (ifull000 < 0) || (ifull000 > ifullmax) ||
            (ifull001 < 0) || (ifull001 > ifullmax) ||
            (ifull010 < 0) || (ifull010 > ifullmax) ||
            (ifull011 < 0) || (ifull011 > ifullmax) ||
            (ifull100 < 0) || (ifull100 > ifullmax) ||
            (ifull101 < 0) || (ifull101 > ifullmax) ||
            (ifull110 < 0) || (ifull110 > ifullmax) ||
            (ifull111 < 0) || (ifull111 > ifullmax)) {
            std::ostringstream o;
            o.precision(16);
            o << "atm_sim_observe : bad full index. "
              << "ifullmax = " << ifullmax << std::endl
              << "ifull000 = " << ifull000 << std::endl
              << "ifull001 = " << ifull001 << std::endl
              << "ifull010 = " << ifull010 << std::endl
              << "ifull011 = " << ifull011 << std::endl
              << "ifull100 = " << ifull100 << std::endl
              << "ifull101 = " << ifull101 << std::endl
              << "ifull110 = " << ifull110 << std::endl
              << "ifull111 = " << ifull111 << std::endl;
            auto & logger = toast::Logger::get();
            logger.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
# endif // ifndef NO_ATM_CHECKS

        int64_t i000 = compressed_index[ifull000];
        int64_t i001 = compressed_index[ifull001];
        int64_t i010 = compressed_index[ifull010];
        int64_t i011 = compressed_index[ifull011];
        int64_t i100 = compressed_index[ifull100];
        int64_t i101 = compressed_index[ifull101];
        int64_t i110 = compressed_index[ifull110];
        int64_t i111 = compressed_index[ifull111];

# ifndef NO_ATM_CHECKS
        int64_t imax = nelem - 1;
        if (
            (i000 < 0) || (i000 > imax) ||
            (i001 < 0) || (i001 > imax) ||
            (i010 < 0) || (i010 > imax) ||
            (i011 < 0) || (i011 > imax) ||
            (i100 < 0) || (i100 > imax) ||
            (i101 < 0) || (i101 > imax) ||
            (i110 < 0) || (i110 > imax) ||
            (i111 < 0) || (i111 > imax)) {
            std::ostringstream o;
            o.precision(16);
            o << "atm_sim_interp : bad compressed index. "
              << "imax = " << imax << std::endl
              << "i000 = " << i000 << std::endl
              << "i001 = " << i001 << std::endl
              << "i010 = " << i010 << std::endl
              << "i011 = " << i011 << std::endl
              << "i100 = " << i100 << std::endl
              << "i101 = " << i101 << std::endl
              << "i110 = " << i110 << std::endl
              << "i111 = " << i111 << std::endl
              << "(x, y, z) = " << x << ", " << y << ", " << z << ")"
              << std::endl
              << "in_cone(x, y, z) = "
              << toast::atm_sim_in_cone(
                x, y, z,
                t_in, delta_t, delta_az, elmin, elmax,
                wx, wy, wz, xstep, ystep, zstep,
                maxdist, cosel0, sinel0
                    );
            auto & logger = toast::Logger::get();
            logger.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
# endif // ifndef NO_ATM_CHECKS

        c000 = realization[i000];
        c001 = realization[i001];
        c010 = realization[i010];
        c011 = realization[i011];
        c100 = realization[i100];
        c101 = realization[i101];
        c110 = realization[i110];
        c111 = realization[i111];

        last_ind[0] = ix;
        last_ind[1] = iy;
        last_ind[2] = iz;

        last_nodes[0] = c000;
        last_nodes[1] = c001;
        last_nodes[2] = c010;
        last_nodes[3] = c011;
        last_nodes[4] = c100;
        last_nodes[5] = c101;
        last_nodes[6] = c110;
        last_nodes[7] = c111;
    } else {
        c000 = last_nodes[0];
        c001 = last_nodes[1];
        c010 = last_nodes[2];
        c011 = last_nodes[3];
        c100 = last_nodes[4];
        c101 = last_nodes[5];
        c110 = last_nodes[6];
        c111 = last_nodes[7];
    }

    double c00 = c000 + (c100 - c000) * dx;
    double c01 = c001 + (c101 - c001) * dx;
    double c10 = c010 + (c110 - c010) * dx;
    double c11 = c011 + (c111 - c011) * dx;

    double c0 = c00 + (c10 - c00) * dy;
    double c1 = c01 + (c11 - c01) * dy;

    double c = c0 + (c1 - c0) * dz;

    return c;
}

int toast::atm_sim_observe(
    size_t nsamp,
    double * times,
    double * az,
    double * el,
    double * tod,
    double T0,
    double azmin,
    double azmax,
    double elmin,
    double elmax,
    double tmin,
    double tmax,
    double rmin,
    double rmax,
    double fixed_r,
    double zatm,
    double zmax,
    double wx,
    double wy,
    double wz,
    double xstep,
    double ystep,
    double zstep,
    double xstart,
    double delta_x,
    double ystart,
    double delta_y,
    double zstart,
    double delta_z,
    double maxdist,
    int64_t nn,
    int64_t nx,
    int64_t ny,
    int64_t nz,
    int64_t xstride,
    int64_t ystride,
    int64_t zstride,
    int64_t nelem,
    int64_t * compressed_index,
    int64_t * full_index,
    double * realization
    ) {
    // For each sample, integrate along the line of sight by summing
    // the atmosphere values. See Church (1995) Section 2.2, first equation.
    // We omit the optical depth factor which is close to unity.

    double zatm_inv = 1.0 / zatm;

    double xstepinv = 1.0 / xstep;
    double ystepinv = 1.0 / ystep;
    double zstepinv = 1.0 / zstep;

    double delta_az = (azmax - azmin);
    double delta_el = (elmax - elmin);
    double delta_t = tmax - tmin;

    double az0 = azmin + delta_az / 2;
    double el0 = elmin + delta_el / 2;
    double sinel0 = sin(el0);
    double cosel0 = cos(el0);

    std::ostringstream o;
    o.precision(16);

    int error = 0;

    # pragma omp parallel
    {
        std::vector <int64_t> last_ind(3);
        std::vector <double> last_nodes(8);

        # pragma omp for schedule(static, 100)
        for (size_t i = 0; i < nsamp; ++i) {
            // # pragma omp flush(error)
            // if (error) continue;

            if ((!((azmin <= az[i]) && (az[i] <= azmax)) &&
                 !((azmin <= az[i] - 2 * M_PI) && (az[i] - 2 * M_PI <= azmax)))
                || !((elmin <= el[i]) && (el[i] <= elmax))) {
            # pragma omp flush(error)
                if (error == 0) {
                    if (atm_verbose()) {
                        o.str("");
                        o <<
                            "atm_sim_observe : observation out of bounds (az, el, t)"
                          << " = (" << az[i] << ",  " << el[i] << ", " << times[i]
                          << ") allowed: (" << azmin << " - " << azmax << ", "
                          << elmin << " - " << elmax << ", "
                          << tmin << " - " << tmax << ")";
                        auto & logger = toast::Logger::get();
                        logger.warning(o.str().c_str());
                    }
                    error = 1;
                # pragma omp flush(error)
                }
                continue;
            }

            double t_now = times[i] - tmin;
            double az_now = az[i] - az0; // Relative to center of field
            double el_now = el[i];

            double xtel_now = wx * t_now;
            double ytel_now = wy * t_now;
            double ztel_now = wz * t_now;

            double sin_el = sin(el_now);
            double sin_el_max = sin(elmax);
            double cos_el = cos(el_now);
            double sin_az = sin(az_now);
            double cos_az = cos(az_now);

            double r = 1.5 * xstep;
            double rstep = xstep;
            while (r < rmin) r += rstep;

            double val = 0;
            if (fixed_r > 0) r = fixed_r;

            while (true) {
                if (r > rmax) break;

                // Coordinates at distance r. The scan is centered on the X-axis

                // Check if the top of the focal plane hits zmax at
                // this distance.  This way all lines-of-sight get
                // integrated to the same distance
                double zz = r * sin_el_max;
                if (zz >= zmax) break;

                // Horizontal coordinates

                zz = r * sin_el;
                double rproj = r * cos_el;
                double xx = rproj * cos_az;
                double yy = rproj * sin_az;

                // Rotate to scan frame

                double x = xx * cosel0 + zz * sinel0;
                double y = yy;
                double z = -xx * sinel0 + zz * cosel0;

                // Translate by the wind

                x += xtel_now;
                y += ytel_now;
                z += ztel_now;

# ifndef NO_ATM_CHECKS
                if ((x < xstart) || (x > xstart + delta_x) ||
                    (y < ystart) || (y > ystart + delta_y) ||
                    (z < zstart) || (z > zstart + delta_z)) {
                #  pragma omp flush (error)
                    if (error == 0) {
                        if (atm_verbose()) {
                            o.str("");
                            o << "atm_sim_observe : (x,y,z) out of bounds: "
                              << std::endl
                              << "x = " << x << std::endl
                              << "y = " << y << std::endl
                              << "z = " << z << std::endl;
                            auto & logger = toast::Logger::get();
                            logger.warning(o.str().c_str());
                        }
                        error = 1;
                    #  pragma omp flush (error)
                    }
                    val = 0;
                    break;
                }
# endif // ifndef NO_ATM_CHECKS

                // Combine atmospheric emission (via interpolation) with the
                // ambient temperature.
                // Note that the r^2 (beam area) and 1/r^2 (source
                // distance) factors cancel in the integral.

                double step_val;
                try {
                    step_val = toast::atm_sim_interp(
                        x, y, z, last_ind, last_nodes,
                        xstart, ystart, zstart,
                        nn, nx, ny, nz,
                        xstride, ystride, zstride,
                        xstep, ystep, zstep,
                        xstepinv, ystepinv, zstepinv,
                        t_now, delta_t, delta_az, elmin, elmax,
                        wx, wy, wz,
                        maxdist, cosel0, sinel0,
                        nelem, compressed_index, full_index,
                        realization
                        ) * (1. - z * zatm_inv);
                } catch (const std::runtime_error & e) {
                    # pragma omp flush(error)
                    if (error == 0) {
                        if (atm_verbose()) {
                            o.str("");
                            o << "atm_sim_observe : interp failed at " << std::endl
                              << "xxyyzz = (" << xx << ", " << yy << ", " << zz <<
                                ")"
                              << std::endl
                              << "xyz = (" << x << ", " << y << ", " << z << ")"
                              << std::endl
                              << "r = " << r << std::endl
                              << "tele at (" << xtel_now << ", " << ytel_now << ", "
                              << ztel_now << ")" << std::endl
                              << "( t, az, el ) = " << "( " << times[i] - tmin << ", "
                              << az_now * 180 / M_PI
                              << " deg , " << el_now * 180 / M_PI << " deg) "
                              << " in_cone(t) = "
                              << toast::atm_sim_in_cone(
                                x, y, z, t_now, delta_t, delta_az, elmin, elmax,
                                wx, wy, wz, xstep, ystep, zstep, maxdist,
                                cosel0, sinel0)
                              << " with "
                              << std::endl << e.what() << std::endl;
                            auto & logger = toast::Logger::get();
                            logger.warning(o.str().c_str());
                        }
                        error = 1;
                    # pragma omp flush(error)
                    }
                    val = 0;
                    break;
                }
                val += step_val;

                // Prepare for the next step

                r += rstep;

                if (fixed_r > 0) break;

                // if ( fixed_r > 0 and r > fixed_r ) break;
            }

            tod[i] = val * rstep * T0;
        }
    }
    return error;
}

#endif // ifdef HAVE_CHOLMOD
