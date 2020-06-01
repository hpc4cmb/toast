
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
#include <algorithm> // std::sort


#ifdef HAVE_CHOLMOD

double median(std::vector <double> vec) {
    if (vec.size() == 0) return 0;

    std::sort(vec.begin(), vec.end());
    int half1 = (vec.size() - 1) * .5;
    int half2 = vec.size() * .5;

    return .5 * (vec[half1] + vec[half2]);
}

double mean(std::vector <double> vec) {
    if (vec.size() == 0) return 0;

    double sum = 0;
    for (auto & val : vec) sum += val;

    return sum / vec.size();
}

toast::atm_sim::atm_sim(double azmin, double azmax, double elmin, double elmax,
                        double tmin, double tmax,
                        double lmin_center, double lmin_sigma,
                        double lmax_center, double lmax_sigma,
                        double w_center, double w_sigma,
                        double wdir_center, double wdir_sigma,
                        double z0_center, double z0_sigma,
                        double T0_center, double T0_sigma,
                        double zatm, double zmax,
                        double xstep, double ystep, double zstep,
                        long nelem_sim_max,
                        int verbosity,
                        uint64_t key1, uint64_t key2,
                        uint64_t counterval1, uint64_t counterval2,
                        std::string cachedir,
                        double rmin, double rmax
                        )
    : cachedir(cachedir),
    verbosity(verbosity),
    key1(key1), key2(key2),
    counter1start(counterval1), counter2start(counterval2),
    azmin(azmin), azmax(azmax),
    elmin(elmin), elmax(elmax), tmin(tmin), tmax(tmax),
    lmin_center(lmin_center), lmin_sigma(lmin_sigma),
    lmax_center(lmax_center), lmax_sigma(lmax_sigma),
    w_center(w_center), w_sigma(w_sigma),
    wdir_center(wdir_center), wdir_sigma(wdir_sigma),
    z0_center(z0_center), z0_sigma(z0_sigma),
    T0_center(T0_center), T0_sigma(T0_sigma),
    zatm(zatm), zmax(zmax),
    xstep(xstep), ystep(ystep), zstep(zstep),
    nelem_sim_max(nelem_sim_max),
    rmin(rmin), rmax(rmax)
{
    counter1 = counter1start;
    counter2 = counter2start;

    corrlim = 1e-3;

    ntask = 1;
    rank = 0;

    auto & env = toast::Environment::get();
    nthread = env.max_threads();

    if ((rank == 0) && (verbosity > 0))
        std::cerr << "atmsim constructed with " << ntask << " processes, "
                  << nthread << " threads per process."
                  << std::endl;

    if (azmin >= azmax) throw std::runtime_error("atmsim: azmin >= azmax.");
    if (elmin < 0) throw std::runtime_error("atmsim: elmin < 0.");
    if (elmax > M_PI_2) throw std::runtime_error("atmsim: elmax > pi/2.");
    if (elmin > elmax) throw std::runtime_error("atmsim: elmin > elmax.");
    if (tmin > tmax) throw std::runtime_error("atmsim: tmin > tmax.");
    if (lmin_center > lmax_center) throw std::runtime_error(
                  "atmsim: lmin_center > lmax_center.");

    xstepinv = 1 / xstep;
    ystepinv = 1 / ystep;
    zstepinv = 1 / zstep;

    delta_az = (azmax - azmin);
    delta_el = (elmax - elmin);
    delta_t = tmax - tmin;

    az0 = azmin + delta_az / 2;
    el0 = elmin + delta_el / 2;
    sinel0 = sin(el0);
    cosel0 = cos(el0);

    xxstep = xstep * cosel0 - zstep * sinel0;
    yystep = ystep;
    zzstep = xstep * sinel0 + zstep * cosel0;

    // speed up the in-cone calculation
    double tol = 0.1 * M_PI / 180; // 0.1 degree tolerance
    tanmin = tan(-0.5 * delta_az - tol);
    tanmax = tan(0.5 * delta_az + tol);

    if ((rank == 0) && (verbosity > 0)) {
        std::cerr << std::endl;
        std::cerr << "Input parameters:" << std::endl;
        std::cerr << "             az = [" << azmin * 180. / M_PI << " - "
                  << azmax * 180. / M_PI << "] (" << delta_az * 180. / M_PI
                  << " degrees)" << std::endl;
        std::cerr << "             el = [" << elmin * 180. / M_PI << " - "
                  << elmax * 180. / M_PI << "] (" << delta_el * 180 / M_PI
                  << " degrees)" << std::endl;
        std::cerr << "              t = [" << tmin << " - " << tmax
                  << "] (" << delta_t << " s)" << std::endl;
        std::cerr << "           lmin = " << lmin_center << " +- " << lmin_sigma
                  << " m" << std::endl;
        std::cerr << "           lmax = " << lmax_center << " +- " << lmax_sigma
                  << " m" << std::endl;
        std::cerr << "              w = " << w_center << " +- " << w_sigma
                  << " m" << std::endl;
        std::cerr << "           wdir = " << wdir_center * 180. / M_PI << " +- "
                  << wdir_sigma * 180. / M_PI << " degrees " << std::endl;
        std::cerr << "             z0 = " << z0_center << " +- " << z0_sigma
                  << " m" << std::endl;
        std::cerr << "             T0 = " << T0_center << " +- " << T0_sigma
                  << " K" << std::endl;
        std::cerr << "           zatm = " << zatm << " m" << std::endl;
        std::cerr << "           zmax = " << zmax << " m" << std::endl;
        std::cerr << "       scan frame: " << std::endl;
        std::cerr << "          xstep = " << xstep << " m" << std::endl;
        std::cerr << "          ystep = " << ystep << " m" << std::endl;
        std::cerr << "          zstep = " << zstep << " m" << std::endl;
        std::cerr << " horizontal frame: " << std::endl;
        std::cerr << "         xxstep = " << xxstep << " m" << std::endl;
        std::cerr << "         yystep = " << yystep << " m" << std::endl;
        std::cerr << "         zzstep = " << zzstep << " m" << std::endl;
        std::cerr << "  nelem_sim_max = " << nelem_sim_max << std::endl;
        std::cerr << "        corrlim = " << corrlim << std::endl;
        std::cerr << "      verbosity = " << verbosity << std::endl;
        std::cerr << "           rmin = " << rmin << " m" << std::endl;
        std::cerr << "           rmax = " << rmax << " m" << std::endl;
    }

    // Initialize cholmod
    chcommon = &cholcommon;
    cholmod_start(chcommon);
    if (verbosity > 1) chcommon->print = 3;  // Default verbosity
    else chcommon->print = 1;                // Minimal verbosity
    chcommon->itype = CHOLMOD_INT;           // All integer arrays are int
    chcommon->dtype = CHOLMOD_DOUBLE;        // All float arrays are double
    chcommon->final_ll = 1;                  // The factorization is LL', not
                                             // LDL'
}

toast::atm_sim::~atm_sim() {
    // Not really necessary- pointer is freed automatically in
    // destructor of unique_ptr.
    compressed_index.reset();
    full_index.reset();
    realization.reset();
    cholmod_finish(chcommon);
}

void toast::atm_sim::print(std::ostream & out) const {
    for (int i = 0; i < ntask; ++i) {
        if (rank != i) continue;
        out << rank << " : cachedir " << cachedir << std::endl;
        out << rank << " : ntask = " << ntask
            << ", nthread = " << nthread << std::endl;
        out << rank << " : verbosity = " << verbosity
            << ", key1 = " << key1
            << ", key2 = " << key2
            << ", counter1 = " << counter1
            << ", counter2 = " << counter2
            << ", counter1start = " << counter1start
            << ", counter2start = " << counter2start << std::endl;
        out << rank << " : azmin = " << azmin
            << ", axmax = " << azmax
            << ", elmin = " << elmin
            << ", elmax = " << elmax
            << ", tmin = " << tmin
            << ", tmax = " << tmax
            << ", sinel0 = " << sinel0
            << ", cosel0 = " << cosel0
            << ", tanmin = " << tanmin
            << ", tanmax = " << tanmax << std::endl;
        out << rank << " : lmin_center = " << lmin_center
            << ", lmax_center = " << lmax_center
            << ", w_center = " << w_center
            << ", w_sigma = " << w_sigma
            << ", wdir_center = " << wdir_center
            << ", wdir_sigma = " << wdir_sigma
            << ", z0_center = " << z0_center
            << ", z0_sigma = " << z0_sigma
            << ", T0_center = " << T0_center
            << ", T0_sigma = " << T0_sigma
            << ", z0inv = " << z0inv << std::endl;
    }
}

void toast::atm_sim::load_realization() {
    cached = false;

    std::ostringstream name;
    name << key1 << "_" << key2 << "_"
         << counter1start << "_" << counter2start;

    char success;

    if (rank == 0) {
        // Load metadata

        success = 1;

        std::ostringstream fname;
        fname << cachedir << "/" << name.str() << "_metadata.txt";

        std::ifstream f(fname.str());
        if (f.good()) {
            f >> nn;
            f >> nelem;
            f >> nx;
            f >> ny;
            f >> nz;
            f >> delta_x;
            f >> delta_y;
            f >> delta_z;
            f >> xstart;
            f >> ystart;
            f >> zstart;
            f >> maxdist;
            f >> wx;
            f >> wy;
            f >> wz;
            f >> lmin;
            f >> lmax;
            f >> w;
            f >> wdir;
            f >> z0;
            f >> T0;
            f.close();

            if (rank == 0 and verbosity > 0) {
                std::cerr << "Loaded metada from "
                          << fname.str() << std::endl;
            }
        } else success = 0;

        if ((verbosity > 0) && success) {
            std::cerr << std::endl;
            std::cerr << "Simulation volume:" << std::endl;
            std::cerr << "   delta_x = " << delta_x << " m" << std::endl;
            std::cerr << "   delta_y = " << delta_y << " m" << std::endl;
            std::cerr << "   delta_z = " << delta_z << " m" << std::endl;
            std::cerr << "    xstart = " << xstart << " m" << std::endl;
            std::cerr << "    ystart = " << ystart << " m" << std::endl;
            std::cerr << "    zstart = " << zstart << " m" << std::endl;
            std::cerr << "   maxdist = " << maxdist << " m" << std::endl;
            std::cerr << "        nx = " << nx << std::endl;
            std::cerr << "        ny = " << ny << std::endl;
            std::cerr << "        nz = " << nz << std::endl;
            std::cerr << "        nn = " << nn << std::endl;
            std::cerr << "Atmospheric realization parameters:" << std::endl;
            std::cerr << " lmin = " << lmin << " m" << std::endl;
            std::cerr << " lmax = " << lmax << " m" << std::endl;
            std::cerr << "    w = " << w << " m/s" << std::endl;
            std::cerr << "   wx = " << wx << " m/s" << std::endl;
            std::cerr << "   wy = " << wy << " m/s" << std::endl;
            std::cerr << "   wz = " << wz << " m/s" << std::endl;
            std::cerr << " wdir = " << wdir * 180. / M_PI << " degrees" << std::endl;
            std::cerr << "   z0 = " << z0 << " m" << std::endl;
            std::cerr << "   T0 = " << T0 << " K" << std::endl;
            std::cerr << "rcorr = " << rcorr << " m (corrlim = "
                      << corrlim << ")" << std::endl;
        }
    }

    if (!success) return;

    zstride = 1;
    ystride = zstride * nz;
    xstride = ystride * ny;

    xstrideinv = 1. / xstride;
    ystrideinv = 1. / ystride;
    zstrideinv = 1. / zstride;

    // Load realization

    try {
        compressed_index.reset(new AlignedVector <long> (nn));
        std::fill(compressed_index->begin(), compressed_index->end(), -1);

        full_index.reset(new AlignedVector <long> (nelem));
        std::fill(full_index->begin(), full_index->end(), -1);
    } catch (...) {
        std::cerr << rank
                  << " : Failed to allocate element indices. nn = "
                  << nn << std::endl;
        throw;
    }
    try {
        realization.reset(new AlignedVector <double> (nelem));
        std::fill(realization->begin(), realization->end(), 0.0);
    } catch (...) {
        std::cerr << rank
                  << " : Failed to allocate realization. nelem = "
                  << nelem << std::endl;
        throw;
    }
    // if (full_index->rank() == 0) {
    std::ostringstream fname_real;
    fname_real << cachedir << "/" << name.str() << "_realization.dat";
    std::ifstream freal(fname_real.str(),
                        std::ios::in | std::ios::binary);

    freal.read((char *)&(*full_index)[0],
               full_index->size() * sizeof(long));
    for (int i = 0; i < nelem; ++i) {
        long ifull = (*full_index)[i];
        (*compressed_index)[ifull] = i;
    }

    freal.read((char *)&(*realization)[0],
               realization->size() * sizeof(double));

    freal.close();

    if (verbosity > 0) std::cerr << "Loaded realization from "
                                 << fname_real.str() << std::endl;

    // }

    cached = true;

    return;
}

void toast::atm_sim::save_realization() {
    if (rank == 0) {
        std::ostringstream name;
        name << key1 << "_" << key2 << "_"
             << counter1start << "_" << counter2start;

        // Save metadata

        std::ostringstream fname;
        fname << cachedir << "/" << name.str() << "_metadata.txt";

        std::ofstream f;
        f.precision(16);
        f.open(fname.str());
        f << nn << std::endl;
        f << nelem << std::endl;
        f << nx << std::endl;
        f << ny << std::endl;
        f << nz << std::endl;
        f << delta_x << std::endl;
        f << delta_y << std::endl;
        f << delta_z << std::endl;
        f << xstart << std::endl;
        f << ystart << std::endl;
        f << zstart << std::endl;
        f << maxdist << std::endl;
        f << wx << std::endl;
        f << wy << std::endl;
        f << wz << std::endl;
        f << lmin << std::endl;
        f << lmax << std::endl;
        f << w << std::endl;
        f << wdir << std::endl;
        f << z0 << std::endl;
        f << T0 << std::endl;
        f.close();

        if (verbosity > 0) std::cerr << "Saved metadata to "
                                     << fname.str() << std::endl;

        // Save realization

        std::ostringstream fname_real;
        fname_real << cachedir << "/" << name.str() << "_realization.dat";
        std::ofstream freal(fname_real.str(),
                            std::ios::out | std::ios::binary);

        freal.write((char *)&(*full_index)[0],
                    full_index->size() * sizeof(long));

        freal.write((char *)&(*realization)[0],
                    realization->size() * sizeof(double));

        freal.close();

        if (verbosity > 0) std::cerr << "Saved realization to "
                                     << fname_real.str() << std::endl;
    }

    return;
}

int toast::atm_sim::simulate(bool use_cache) {
    if (use_cache) load_realization();

    if (cached) return 0;

    try {
        draw();

        get_volume();

        compress_volume();

        if (rank == 0 and verbosity > 0) {
            std::cerr << "Resizing realization to " << nelem << std::endl;
        }

        try {
            realization.reset(new AlignedVector <double> (nelem));
            std::fill(realization->begin(), realization->end(), 0.0);
        } catch (...) {
            std::cerr << rank
                      << " : Failed to allocate realization. nelem = "
                      << nelem << std::endl;
            throw;
        }
        toast::Timer tm;
        tm.start();

        long ind_start = 0, ind_stop = 0, slice = 0;

        // Simulate the atmosphere in independent slices, each slice
        // assigned to one process

        std::vector <int> slice_starts;
        std::vector <int> slice_stops;

        while (true) {
            get_slice(ind_start, ind_stop);
            slice_starts.push_back(ind_start);
            slice_stops.push_back(ind_stop);

            if (slice % ntask == rank) {
                cholmod_sparse * cov = build_sparse_covariance(ind_start,
                                                               ind_stop);
                cholmod_sparse * sqrt_cov = sqrt_sparse_covariance(cov,
                                                                   ind_start,
                                                                   ind_stop);
                cholmod_free_sparse(&cov, chcommon);
                apply_sparse_covariance(sqrt_cov,
                                        ind_start,
                                        ind_stop);
                cholmod_free_sparse(&sqrt_cov, chcommon);
            }

            // Advance the RNG counter on all processes
            counter2 += ind_stop - ind_start;

            if (ind_stop == nelem) break;

            ++slice;
        }

        // smooth();

        tm.stop();
        if ((rank == 0) && (verbosity > 0)) {
            tm.report("Realization constructed in");
        }
    } catch (const std::exception & e) {
        std::cerr << "WARNING: atm::simulate failed with: " << e.what()
                  << std::endl;
    }
    cached = true;

    if (use_cache) save_realization();

    return 0;
}

void toast::atm_sim::get_slice(long & ind_start, long & ind_stop) {
    // Identify a manageable slice of compressed indices to simulate next

    // Move element counter to the end of the most recent simulated slice
    ind_start = ind_stop;

    long ix_start = (*full_index)[ind_start] * xstrideinv;
    long ix1 = ix_start;
    long ix2;

    while (true) {
        // Advance element counter by one layer of elements
        ix2 = ix1;
        while (ix1 == ix2) {
            ++ind_stop;
            if (ind_stop == nelem) break;
            ix2 = (*full_index)[ind_stop] * xstrideinv;
        }

        // Check if there are no more elements
        if (ind_stop == nelem) break;

        // Check if we have enough to meet the minimum number of elements
        if (ind_stop - ind_start >= nelem_sim_max) break;

        // Check if we have enough layers
        // const int nlayer_sim_max = 10;
        // if ( ix2 - ix_start >= nlayer_sim_max ) break;
        ix1 = ix2;
    }

    if ((rank == 0) && (verbosity > 0)) {
        std::cerr << "X-slice: " << ix_start * xstep << " -- " << ix2 * xstep
                  << "(" << ix2 - ix_start <<  " " << xstep << " m layers)"
                  << " m out of  " << nx * xstep << " m"
                  << " indices " << ind_start << " -- " << ind_stop
                  << " ( " << ind_stop - ind_start << " )"
                  << " out of " << nelem << std::endl;
    }

    return;
}

void toast::atm_sim::smooth() {
    // Replace each vertex with a mean of its immediate vicinity

    toast::Timer tm;
    tm.start();

    double coord[3];

    std::vector <double> smoothed_realization(realization->size());

    for (size_t i = 0; i < full_index->size(); ++i) {
        ind2coord(i, coord);
        long ix = coord[0] * xstepinv;
        long iy = coord[1] * ystepinv;
        long iz = coord[2] * zstepinv;

        long offset = ix * xstride + iy * ystride + iz * zstride;

        long w = 3; // width of the smoothing kernel
        long ifullmax = compressed_index->size();

        std::vector <double> vals;

        // for (int xoff=-w; xoff <= w; ++xoff) {
        for (int xoff = 0; xoff <= 0; ++xoff) {
            if (ix + xoff < 0) continue;
            if (ix + xoff >= nx) break;

            for (int yoff = -w; yoff <= w; ++yoff) {
                if (iy + yoff < 0) continue;
                if (iy + yoff >= ny) break;

                for (int zoff = -w; zoff <= w; ++zoff) {
                    if (iz + zoff < 0) continue;
                    if (iz + zoff >= nz) break;

                    long ifull = offset + xoff * xstride + yoff * ystride
                                 + zoff * zstride;

                    if ((ifull < 0) || (ifull >= ifullmax)) throw std::runtime_error(
                                  "Index out of range in smoothing.");

                    long ii = (*compressed_index)[ifull];

                    if (ii >= 0) {
                        vals.push_back((*realization)[ii]);
                    }
                }
            }
        }

        // Get the smoothed value

        smoothed_realization[i] = mean(vals);
    }

    // if (realization->rank() == 0) {
    for (int i = 0; i < realization->size(); ++i) {
        (*realization)[i] = smoothed_realization[i];
    }

    // }

    tm.stop();

    if ((rank == 0) && (verbosity > 0)) {
        tm.report("Realization smoothed in");
    }
    return;
}

int toast::atm_sim::observe(double * t, double * az, double * el, double * tod,
                            long nsamp, double fixed_r) {
    if (!cached) {
        throw std::runtime_error("There is no cached observation to observe");
    }

    toast::Timer tm;
    tm.start();

    // For each sample, integrate along the line of sight by summing
    // the atmosphere values. See Church (1995) Section 2.2, first equation.
    // We omit the optical depth factor which is close to unity.

    double zatm_inv = 1. / zatm;

    std::ostringstream o;
    o.precision(16);
    int error = 0;

    # pragma omp parallel for schedule(static, 100)
    for (long i = 0; i < nsamp; ++i) {
        # pragma omp flush(error)
        if (error) continue;

        if ((!((azmin <= az[i]) && (az[i] <= azmax)) &&
             !((azmin <= az[i] - 2 * M_PI) && (az[i] - 2 * M_PI <= azmax)))
            || !((elmin <= el[i]) && (el[i] <= elmax))) {
            o.precision(16);
            o << "atmsim::observe : observation out of bounds (az, el, t)"
              << " = (" << az[i] << ",  " << el[i] << ", " << t[i]
              << ") allowed: (" << azmin << " - " << azmax << ", "
              << elmin << " - " << elmax << ", "
              << tmin << " - " << tmax << ")"
              << std::endl;
            error = 1;
            # pragma omp flush(error)
            continue;
        }

        double t_now = t[i] - tmin;
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

        std::vector <long> last_ind(3);
        std::vector <double> last_nodes(8);

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
                o << "atmsim::observe : (x,y,z) out of bounds: "
                  << std::endl
                  << "x = " << x << std::endl
                  << "y = " << y << std::endl
                  << "z = " << z << std::endl;
                error = 1;
                #  pragma omp flush (error)
                break;
            }
# endif // ifndef NO_ATM_CHECKS

            // Combine atmospheric emission (via interpolation) with the
            // ambient temperature.
            // Note that the r^2 (beam area) and 1/r^2 (source
            // distance) factors cancel in the integral.

            double step_val;
            try {
                step_val = interp(x, y, z, last_ind, last_nodes)
                           * (1. - z * zatm_inv);
            } catch (const std::runtime_error & e) {
                std::ostringstream o;
                o << "atmsim::observe : interp failed at " << std::endl
                  << "xxyyzz = (" << xx << ", " << yy << ", " << zz << ")"
                  << std::endl
                  << "xyz = (" << x << ", " << y << ", " << z << ")"
                  << std::endl
                  << "r = " << r << std::endl
                  << "tele at (" << xtel_now << ", " << ytel_now << ", "
                  << ztel_now << ")" << std::endl
                  << "( t, az, el ) = " << "( " << t[i] - tmin << ", "
                  << az_now * 180 / M_PI
                  << " deg , " << el_now * 180 / M_PI << " deg) "
                  << " in_cone(t) = " << in_cone(x, y, z, t_now)
                  << " with "
                  << std::endl << e.what() << std::endl;
                error = 1;
                # pragma omp flush(error)
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

    tm.stop();

    if ((rank == 0) && (verbosity > 0)) {
        if (fixed_r > 0) {
            std::ostringstream o;
            o << " samples observed at r =  " << fixed_r << " in";
            tm.report(o.str().c_str());
        } else {
            tm.report(" samples observed in");
        }
    }

    if (error) {
        std::cerr << "WARNING: atm::observe failed with: \"" << o.str()
                  << "\"" << std::endl;
        return -1;
    }

    return 0;
}

void toast::atm_sim::draw() {
    // Draw 10000 gaussian variates to use in drawing the simulation
    // parameters

    const size_t nrand = 10000;
    double randn[nrand];
    toast::rng_dist_normal(nrand, key1, key2, counter1, counter2, randn);
    counter2 += nrand;
    double * prand = randn;
    long irand = 0;

    if (rank == 0) {
        lmin = 0;
        lmax = 0;
        w = -1;
        wdir = 0;
        z0 = 0;
        T0 = 0;

        while (lmin >= lmax) {
            lmin = 0;
            lmax = 0;
            while (lmin <= 0 &&
                   irand < nrand - 1) lmin = lmin_center + randn[irand++] * lmin_sigma;
            while (lmax <= 0 &&
                   irand < nrand - 1) lmax = lmax_center + randn[irand++] * lmax_sigma;
        }
        while (w < 0 && irand < nrand - 1) w = w_center + randn[irand++] * w_sigma;
        wdir = fmod(wdir_center + randn[irand++] * wdir_sigma, M_PI);
        while (z0 <= 0 && irand < nrand) z0 = z0_center + randn[irand++] * z0_sigma;
        while (T0 <= 0 && irand < nrand) T0 = T0_center + randn[irand++] * T0_sigma;

        if (irand == nrand) throw std::runtime_error(
                      "Failed to draw parameters so satisfy boundary conditions");
    }

    // Precalculate the ratio for covariance

    z0inv = 1. / (2. * z0);

    // Wind is parallel to surface. Rotate to a frame where the scan
    // is across the X-axis.

    double eastward_wind = w * cos(wdir);
    double northward_wind = w * sin(wdir);

    double angle = az0 - M_PI / 2;
    double wx_h = eastward_wind * cos(angle) - northward_wind * sin(angle);
    wy = eastward_wind * sin(angle) + northward_wind * cos(angle);

    wx = wx_h * cosel0;
    wz = -wx_h * sinel0;

    // Inverse the wind direction so we can apply it to the
    // telescope position

    wx = -wx;
    wy = -wy;

    if ((rank == 0) && (verbosity > 0)) {
        std::cerr << std::endl;
        std::cerr << "Atmospheric realization parameters:" << std::endl;
        std::cerr << " lmin = " << lmin << " m" << std::endl;
        std::cerr << " lmax = " << lmax << " m" << std::endl;
        std::cerr << "    w = " << w << " m/s" << std::endl;
        std::cerr << " easthward wind = " << eastward_wind << " m/s" << std::endl;
        std::cerr << " northward wind = " << northward_wind << " m/s" << std::endl;
        std::cerr << "  az0 = " << az0 * 180. / M_PI << " degrees" << std::endl;
        std::cerr << "  el0 = " << el0 * 180. / M_PI << " degrees" << std::endl;
        std::cerr << "   wx = " << wx << " m/s" << std::endl;
        std::cerr << "   wy = " << wy << " m/s" << std::endl;
        std::cerr << "   wz = " << wz << " m/s" << std::endl;
        std::cerr << " wdir = " << wdir * 180. / M_PI << " degrees" << std::endl;
        std::cerr << "   z0 = " << z0 << " m" << std::endl;
        std::cerr << "   T0 = " << T0 << " K" << std::endl;
    }

    return;
}

void toast::atm_sim::get_volume() {
    // Trim zmax if rmax sets a more stringent limit

    double zmax_test = rmax * sin(elmax);
    if (zmax > zmax_test) {
        zmax = zmax_test;
    }

    // Horizontal volume

    double delta_z_h = zmax;

    // std::cerr << "delta_z_h = " << delta_z_h << std::endl;

    // Maximum distance observed through the simulated volume
    maxdist = delta_z_h / sinel0;

    // std::cerr << "maxdist = " << maxdist << std::endl;

    // Volume length
    double delta_x_h = maxdist * cos(elmin);

    // std::cerr << "delta_x_h = " << delta_x_h << std::endl;

    double x, y, z, xx, zz, r, rproj, z_min, z_max;
    r = maxdist;

    z = r * sin(elmin);
    rproj = r * cos(elmin);
    x = rproj * cos(0);
    z_min = -x * sinel0 + z * cosel0;

    z = r * sin(elmax);
    rproj = r * cos(elmax);
    x = rproj * cos(delta_az / 2);
    z_max = -x * sinel0 + z * cosel0;

    // Cone width
    rproj = r * cos(elmin);
    if (delta_az > M_PI) delta_y_cone = 2 * rproj;
    else delta_y_cone = 2 * rproj * cos(0.5 * (M_PI - delta_az));

    // std::cerr << "delta_y_cone = " << delta_y_cone << std::endl;

    // Cone height
    delta_z_cone = z_max - z_min;

    // std::cerr << "delta_z_cone = " << delta_z_cone << std::endl;

    // Rotate to observation plane

    delta_x = maxdist;

    // std::cerr << "delta_x = " << delta_x << std::endl;
    delta_z = delta_z_cone;

    // std::cerr << "delta_z = " << delta_z << std::endl;

    // Wind effect

    double wdx = std::abs(wx) * delta_t;
    double wdy = std::abs(wy) * delta_t;
    double wdz = std::abs(wz) * delta_t;
    delta_x += wdx;
    delta_y = delta_y_cone + wdy;
    delta_z += wdz;

    // Margin for interpolation

    delta_x += xstep;
    delta_y += 2 * ystep;
    delta_z += 2 * zstep;

    // Translate the volume to allow for wind.  Telescope sits
    // at (0, 0, 0) at t=0

    if (wx < 0) xstart = -wdx;
    else xstart = 0;

    if (wy < 0) ystart = -0.5 * delta_y_cone - wdy - ystep;
    else ystart = -0.5 * delta_y_cone - ystep;

    if (wz < 0) zstart = z_min - wdz - zstep;
    else zstart = z_min - zstep;

    // Grid points

    nx = delta_x / xstep + 1;
    ny = delta_y / ystep + 1;
    nz = delta_z / zstep + 1;
    nn = nx * ny * nz;

    // 1D storage of the 3D volume elements

    zstride = 1;
    ystride = zstride * nz;
    xstride = ystride * ny;

    xstrideinv = 1. / xstride;
    ystrideinv = 1. / ystride;
    zstrideinv = 1. / zstride;

    if ((rank == 0) && (verbosity > 0)) {
        std::cerr << std::endl;
        std::cerr << "Simulation volume:" << std::endl;
        std::cerr << "   delta_x = " << delta_x << " m" << std::endl;
        std::cerr << "   delta_y = " << delta_y << " m" << std::endl;
        std::cerr << "   delta_z = " << delta_z << " m" << std::endl;
        std::cerr << "Observation cone along the X-axis:" << std::endl;
        std::cerr << "   delta_y_cone = " << delta_y_cone << " m" << std::endl;
        std::cerr << "   delta_z_cone = " << delta_z_cone << " m" << std::endl;
        std::cerr << "    xstart = " << xstart << " m" << std::endl;
        std::cerr << "    ystart = " << ystart << " m" << std::endl;
        std::cerr << "    zstart = " << zstart << " m" << std::endl;
        std::cerr << "   maxdist = " << maxdist << " m" << std::endl;
        std::cerr << "        nx = " << nx << std::endl;
        std::cerr << "        ny = " << ny << std::endl;
        std::cerr << "        nz = " << nz << std::endl;
        std::cerr << "        nn = " << nn << std::endl;
    }

    initialize_kolmogorov();
}

void toast::atm_sim::initialize_kolmogorov() {
    auto & logger = toast::Logger::get();
    toast::Timer tm;
    tm.start();

    // Numerically integrate the modified Kolmogorov spectrum for the
    // correlation function at grid points. We integrate down from
    // 10*kappamax to 0 for numerical precision

    rmin_kolmo = 0;
    double diag = sqrt(delta_x * delta_x + delta_y * delta_y);
    rmax_kolmo = sqrt(diag * diag + delta_z * delta_z) * 1.01;
    nr = 1000; // Size of the interpolation grid

    rstep = (rmax_kolmo - rmin_kolmo) / (nr - 1);
    rstep_inv = 1. / rstep;

    kolmo_x.clear();
    kolmo_x.resize(nr, 0);
    kolmo_y.clear();
    kolmo_y.resize(nr, 0);

    double kappamin = 1. / lmax;
    double kappamax = 1. / lmin;
    double kappal = 0.9 * kappamax;
    double invkappal = 1 / kappal;     // Optimize
    double kappa0 = 0.75 * kappamin;
    double kappa0sq = kappa0 * kappa0; // Optimize
    long nkappa = 10000;               // Number of integration steps
    double kappastart = 1e-4;
    double kappastop = 10 * kappamax;

    // kappa = exp(ikappa * kappascale) * kappastart
    double kappascale = log(kappastop / kappastart) / (nkappa - 1);

    double slope1 = 7. / 6.;
    double slope2 = -11. / 6.;

    if ((rank == 0) && (verbosity > 0)) {
        std::cerr << std::endl;
        std::cerr << "Evaluating Kolmogorov correlation at " << nr
                  << " different separations in range " << rmin_kolmo
                  << " - " << rmax_kolmo << " m" << std::endl;
        std::cerr << "kappamin = " << kappamin
                  << " 1/m, kappamax =  " << kappamax
                  << " 1/m. nkappa = " << nkappa << std::endl;
    }

    // Use Newton's method to integrate the correlation function

    long nkappa_task = nkappa / ntask + 1;
    long first_kappa = nkappa_task * rank;
    long last_kappa = first_kappa + nkappa_task;
    if (last_kappa > nkappa) last_kappa = nkappa;

    // Precalculate the power spectrum function

    std::vector <double> phi(last_kappa - first_kappa);
    std::vector <double> kappa(last_kappa - first_kappa);
    # pragma omp parallel for schedule(static, 10)
    for (long ikappa = first_kappa; ikappa < last_kappa; ++ikappa) {
        kappa[ikappa - first_kappa] = exp(ikappa * kappascale) * kappastart;
    }

    # pragma omp parallel for schedule(static, 10)
    for (long ikappa = first_kappa; ikappa < last_kappa; ++ikappa) {
        double k = exp(ikappa * kappascale) * kappastart;
        double kkl = k * invkappal;
        phi[ikappa - first_kappa] =
            (1. + 1.802 * kkl - 0.254 * pow(kkl, slope1))
            * exp(-kkl * kkl) * pow(k * k + kappa0sq, slope2);
    }

    if ((rank == 0) && (verbosity > 0)) {
        std::ofstream f;
        std::ostringstream fname;
        fname << "kolmogorov_f.txt";
        f.open(fname.str(), std::ios::out);
        for (int ikappa = 0; ikappa < nkappa; ++ikappa) {
            f << kappa[ikappa] << " " << phi[ikappa] << std::endl;
        }
        f.close();
    }

    // Newton's method factors, not part of the power spectrum

    if (first_kappa == 0) phi[0] /= 2;
    if (last_kappa == nkappa) phi[last_kappa - first_kappa - 1] /= 2;

    // Integrate the power spectrum for a spherically symmetric
    // correlation function

    double nri = 1. / (nr - 1);
    double tau = 10.;
    double enorm = 1. / (exp(tau) - 1.);
    double ifac3 = 1. / (2. * 3.);

    # pragma omp parallel for schedule(static, 10)
    for (long ir = 0; ir < nr; ++ir) {
        double r = rmin_kolmo
                   + (exp(ir * nri * tau) - 1) * enorm * (rmax_kolmo - rmin_kolmo);
        double rinv = 1 / r;
        double val = 0;
        if (r * kappamax < 1e-2) {
            // special limit r -> 0,
            // sin(kappa.r)/r -> kappa - kappa^3*r^2/3!
            double r2 = r * r;
            for (long ikappa = first_kappa; ikappa < last_kappa - 1; ++ikappa) {
                double k = kappa[ikappa - first_kappa];
                double kstep = kappa[ikappa + 1 - first_kappa] - k;
                double kappa2 = k * k;
                double kappa4 = kappa2 * kappa2;
                val += phi[ikappa - first_kappa] * (kappa2 - r2 * kappa4 * ifac3) *
                       kstep;
            }
        } else {
            for (long ikappa = first_kappa; ikappa < last_kappa - 1; ++ikappa) {
                double k1 = kappa[ikappa - first_kappa];
                double k2 = kappa[ikappa + 1 - first_kappa];
                double phi1 = phi[ikappa - first_kappa];
                double phi2 = phi[ikappa + 1 - first_kappa];
                val += .5 * (phi1 + phi2) * rinv *
                       (k1 * cos(k1 * r) - k2 * cos(k2 * r)
                        - rinv * (sin(k1 * r) - sin(k2 * r))
                       );
            }
            val /= r;
        }
        kolmo_x[ir] = r;
        kolmo_y[ir] = val;
    }

    // Normalize

    double norm = 1. / kolmo_y[0];
    for (int i = 0; i < nr; ++i) kolmo_y[i] *= norm;

    if ((rank == 0) && (verbosity > 0)) {
        std::ofstream f;
        std::ostringstream fname;
        fname << "kolmogorov.txt";
        f.open(fname.str(), std::ios::out);
        for (int ir = 0; ir < nr;
             ir++) f << kolmo_x[ir] << " " << kolmo_y[ir] << std::endl;
        f.close();
    }

    // Measure the correlation length
    long icorr = nr - 1;
    while (fabs(kolmo_y[icorr]) < corrlim) --icorr;
    rcorr = kolmo_x[icorr];
    rcorrsq = rcorr * rcorr;

    tm.stop();

    if ((rank == 0) && (verbosity > 0)) {
        std::ostringstream o;
        o << "rcorr = " << rcorr << " m (corrlim = " << corrlim << ")";
        logger.debug(o.str().c_str());
        tm.report("Kolmogorov initialized in");
    }

    return;
}

double toast::atm_sim::kolmogorov(double r) {
    // Return autocovariance of a Kolmogorov process at separation r

    if (r == 0) return kolmo_y[0];

    if (r == rmax_kolmo) return kolmo_y[nr - 1];

    if ((r < rmin_kolmo) || (r > rmax_kolmo)) {
        std::ostringstream o;
        o.precision(16);
        o << "Kolmogorov value requested at " << r
          << ", outside gridded range [" << rmin_kolmo << ", " << rmax_kolmo << "].";
        throw std::runtime_error(o.str().c_str());
    }

    // Simple linear interpolation for now.  Use a bisection method
    // to find the rigth elements.

    long low = 0, high = nr - 1;
    long ir;

    while (true) {
        ir = low + 0.5 * (high - low);
        if (kolmo_x[ir] <= r and r <= kolmo_x[ir + 1]) break;
        if (r < kolmo_x[ir]) high = ir;
        else low = ir;
    }

    double rlow = kolmo_x[ir];
    double rhigh = kolmo_x[ir + 1];
    double rdist = (r - rlow) / (rhigh - rlow);
    double vlow = kolmo_y[ir];
    double vhigh = kolmo_y[ir + 1];

    double val = (1 - rdist) * vlow + rdist * vhigh;

    return val;
}

void toast::atm_sim::compress_volume() {
    // Establish a mapping between full volume indices and observed
    // volume indices
    toast::Timer tm;
    tm.start();

    if ((rank == 0) && (verbosity > 0)) {
        std::cerr << "Compressing volume, N = " << nn << std::endl;
    }

    std::vector <unsigned char> hit;
    try {
        compressed_index.reset(new AlignedVector <long> (nn));
        std::fill(compressed_index->begin(), compressed_index->end(), -1);

        full_index.reset(new AlignedVector <long> (nn));
        std::fill(full_index->begin(), full_index->end(), -1);

        hit.resize(nn, false);
    } catch (...) {
        std::cerr << rank
                  << " : Failed to allocate element indices. nn = "
                  << nn << std::endl;
        throw;
    }
    // Start by flagging all elements that are hit

    for (long ix = 0; ix < nx - 1; ++ix) {
        if (ix % ntask != rank) continue;
        double x = xstart + ix * xstep;

        # pragma omp parallel for schedule(static, 10)
        for (long iy = 0; iy < ny - 1; ++iy) {
            double y = ystart + iy * ystep;

            for (long iz = 0; iz < nz - 1; ++iz) {
                double z = zstart + iz * zstep;
                if (in_cone(x, y, z)) {
# ifndef NO_ATM_CHECKS
                    hit.at(ix * xstride + iy * ystride + iz * zstride) = true;
# else // ifndef NO_ATM_CHECKS
                    hit[ix * xstride + iy * ystride + iz * zstride] = true;
# endif // ifndef NO_ATM_CHECKS
                }
            }
        }
    }

    if ((rank == 0) && (verbosity > 0)) {
        std::cerr << "Flagged hits, flagging neighbors" << std::endl;
    }

    // For extra margin, flag all the neighbors of the hit elements

    std::vector <unsigned char> hit2 = hit;

    for (long ix = 1; ix < nx - 1; ++ix) {
        if (ix % ntask != rank) continue;

        # pragma omp parallel for schedule(static, 10)
        for (long iy = 1; iy < ny - 1; ++iy) {
            for (long iz = 1; iz < nz - 1; ++iz) {
                long offset = ix * xstride + iy * ystride + iz * zstride;

                if (hit2[offset]) {
                    // Flag this element but also its neighbours to facilitate
                    // interpolation

                    for (double xmul = -2; xmul < 4; ++xmul) {
                        if ((ix + xmul < 0) || (ix + xmul > nx - 1)) continue;

                        for (double ymul = -2; ymul < 4; ++ymul) {
                            if ((iy + ymul < 0) || (iy + ymul > ny - 1)) continue;

                            for (double zmul = -2; zmul < 4; ++zmul) {
                                if ((iz + zmul < 0) || (iz + zmul > nz - 1)) continue;

# ifndef NO_ATM_CHECKS
                                hit.at(offset + xmul * xstride
                                       + ymul * ystride + zmul * zstride) = true;
# else // ifndef NO_ATM_CHECKS
                                hit[offset + xmul * xstride
                                    + ymul * ystride + zmul * zstride] = true;
# endif // ifndef NO_ATM_CHECKS
                            }
                        }
                    }
                }
            }
        }
    }

    hit2.resize(0);

    if ((rank == 0) && (verbosity > 0)) {
        std::cerr << "Creating compression table" << std::endl;
    }

    // Then create the mappings between the compressed and full indices

    long i = 0;
    for (long ifull = 0; ifull < nn; ++ifull) {
        if (hit[ifull]) {
            (*full_index)[i] = ifull;
            (*compressed_index)[ifull] = i;
            ++i;
        }
    }

    hit.resize(0);
    nelem = i;

    full_index->resize(nelem);

    tm.stop();

    if (rank == 0) {
        // if ( verbosity > 0 ) {
        tm.report("Volume compressed in");
        std::cout << i << " / " << nn << "(" << i * 100. / nn << " %)"
                  << " volume elements are needed for the simulation"
                  << std::endl
                  << "nx = " << nx << " ny = " << ny << " nz = " << nz
                  << std::endl
                  << "wx = " << wx << " wy = " << wy << " wz = " << wz
                  << std::endl;

        // }
    }

    if (nelem == 0) throw std::runtime_error("No elements in the observation cone.");
}

bool toast::atm_sim::in_cone(double x, double y, double z, double t_in) {
    // Input coordinates are in the scan frame, rotate to horizontal frame

    double tstep = 1;

    for (double t = 0; t < delta_t; t += tstep) {
        if (t_in >= 0) {
            if (t != 0) break;
            t = t_in;
        }

        if ((t_in < 0) && (delta_t - t < tstep)) t = delta_t;

        double xtel_now = wx * t;
        double dx = x - xtel_now;

        // Is the point behind the telescope at this time?

        if (dx + xstep < 0) {
            if (t_in >= 0) std::cerr << "dx + xstep < 0: " << dx << std::endl;
            continue;
        }

        // Check the rest of the spherical coordinates

        double ytel_now = wy * t;
        double dy = y - ytel_now;

        double ztel_now = wz * t;
        double dz = z - ztel_now;

        double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (r > maxdist * 1.01) {
            if (t_in >= 0) std::cerr << "r > maxdist " << r << std::endl;
            continue;
        }

        if (dz > 0) dz -= zstep;
        else dz += zstep;

        if ((std::abs(dy) < 2 * ystep) && (std::abs(dz) < 2 * zstep)) return true;

        double dxx = dx * cosel0 - dz * sinel0;
        double dyy = dy;
        double dzz = dx * sinel0 + dz * cosel0;

        double el = std::asin(dzz / r);
        if ((el < elmin) || (el > elmax)) {
            if (t_in >= 0)
                std::cerr << "el outside cone: "
                          << el * 180 / M_PI << " not in "
                          << elmin * 180 / M_PI << " - "
                          << elmax * 180 / M_PI << std::endl;
            continue;
        }

        dxx = (dx + xstep) * cosel0 - dz * sinel0;
        double az = std::atan2(dyy, dxx);
        if (std::abs(az) > 0.5 * delta_az) {
            if (t_in >= 0)
                std::cerr << "abs(az) > delta_az/2 "
                          << az * 180 / M_PI << " > "
                          << 0.5 * delta_az * 180 / M_PI << std::endl;
            continue;
        }

        // Passed all the checks

        return true;
    }

    return false;
}

void toast::atm_sim::ind2coord(long i, double * coord) {
    // Translate a compressed index into xyz-coordinates
    // in the horizontal frame

    long ifull = (*full_index)[i];

    long ix = ifull * xstrideinv;
    long iy = (ifull - ix * xstride) * ystrideinv;
    long iz = ifull - ix * xstride - iy * ystride;

    // coordinates in the scan frame

    double x = xstart + ix * xstep;
    double y = ystart + iy * ystep;
    double z = zstart + iz * zstep;

    // Into the horizontal frame

    coord[0] = x * cosel0 - z * sinel0;
    coord[1] = y;
    coord[2] = x * sinel0 + z * cosel0;
}

long toast::atm_sim::coord2ind(double x, double y, double z) {
    // Translate scan frame xyz-coordinates into a compressed index

    long ix = (x - xstart) * xstepinv;
    long iy = (y - ystart) * ystepinv;
    long iz = (z - zstart) * zstepinv;

# ifndef NO_ATM_CHECKS
    if ((ix < 0) || (ix > nx - 1) || (iy < 0) || (iy > ny - 1) || (iz < 0) ||
        (iz > nz - 1)) {
        std::ostringstream o;
        o.precision(16);
        o << "atmsim::coord2ind : full index out of bounds at ("
          << x << ", " << y << ", " << z << ") = ("
          << ix << " /  " << nx << ", " << iy << " / " << ny << ", "
          << iz << ", " << nz << ")";
        throw std::runtime_error(o.str().c_str());
    }
# endif // ifndef NO_ATM_CHECKS

    size_t ifull = ix * xstride + iy * ystride + iz * zstride;

    return (*compressed_index)[ifull];
}

double toast::atm_sim::interp(double x, double y, double z,
                              std::vector <long> & last_ind,
                              std::vector <double> & last_nodes) {
    // Trilinear interpolation

    long ix = (x - xstart) * xstepinv;
    long iy = (y - ystart) * ystepinv;
    long iz = (z - zstart) * zstepinv;

    double dx = (x - (xstart + (double)ix * xstep)) * xstepinv;
    double dy = (y - (ystart + (double)iy * ystep)) * ystepinv;
    double dz = (z - (zstart + (double)iz * zstep)) * zstepinv;

# ifndef NO_ATM_CHECKS
    if ((dx < 0) || (dx > 1) || (dy < 0) || (dy > 1) || (dz < 0) || (dz > 1)) {
        std::ostringstream o;
        o.precision(16);
        o << "atmsim::interp : bad fractional step: " << std::endl
          << "x = " << x << std::endl
          << "y = " << y << std::endl
          << "z = " << z << std::endl
          << "dx = " << dx << std::endl
          << "dy = " << dy << std::endl
          << "dz = " << dz << std::endl;
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
            o << "atmsim::interp : full index out of bounds at"
              << std::endl << "("
              << x << ", " << y << ", " << z << ") = ("
              << ix << "/" << nx << ", "
              << iy << "/" << ny << ", "
              << iz << "/" << nz << ")";
            throw std::runtime_error(o.str().c_str());
        }
# endif // ifndef NO_ATM_CHECKS

        size_t offset = ix * xstride + iy * ystride + iz * zstride;

        size_t ifull000 = offset;
        size_t ifull001 = offset + zstride;
        size_t ifull010 = offset + ystride;
        size_t ifull011 = ifull010 + zstride;
        size_t ifull100 = offset + xstride;
        size_t ifull101 = ifull100 + zstride;
        size_t ifull110 = ifull100 + ystride;
        size_t ifull111 = ifull110 + zstride;

# ifndef NO_ATM_CHECKS
        long ifullmax = compressed_index->size() - 1;
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
            o << "atmsim::observe : bad full index. "
              << "ifullmax = " << ifullmax << std::endl
              << "ifull000 = " << ifull000 << std::endl
              << "ifull001 = " << ifull001 << std::endl
              << "ifull010 = " << ifull010 << std::endl
              << "ifull011 = " << ifull011 << std::endl
              << "ifull100 = " << ifull100 << std::endl
              << "ifull101 = " << ifull101 << std::endl
              << "ifull110 = " << ifull110 << std::endl
              << "ifull111 = " << ifull111 << std::endl;
            throw std::runtime_error(o.str().c_str());
        }
# endif // ifndef NO_ATM_CHECKS

        long i000 = (*compressed_index)[ifull000];
        long i001 = (*compressed_index)[ifull001];
        long i010 = (*compressed_index)[ifull010];
        long i011 = (*compressed_index)[ifull011];
        long i100 = (*compressed_index)[ifull100];
        long i101 = (*compressed_index)[ifull101];
        long i110 = (*compressed_index)[ifull110];
        long i111 = (*compressed_index)[ifull111];

# ifndef NO_ATM_CHECKS
        long imax = realization->size() - 1;
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
            o << "atmsim::observe : bad compressed index. "
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
              << "in_cone(x, y, z) = " << in_cone(x, y, z)
              << std::endl;
            throw std::runtime_error(o.str().c_str());
        }
# endif // ifndef NO_ATM_CHECKS

        c000 = (*realization)[i000];
        c001 = (*realization)[i001];
        c010 = (*realization)[i010];
        c011 = (*realization)[i011];
        c100 = (*realization)[i100];
        c101 = (*realization)[i101];
        c110 = (*realization)[i110];
        c111 = (*realization)[i111];

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

cholmod_sparse * toast::atm_sim::build_sparse_covariance(long ind_start,
                                                         long ind_stop) {
    // Build a sparse covariance matrix.

    toast::Timer tm;
    tm.start();

    // Build the covariance matrix first in the triplet form, then
    // cast it to the column-packed format.

    std::vector <int> rows, cols;
    std::vector <double> vals;
    size_t nelem = ind_stop - ind_start; // Number of elements in the slice
    std::vector <double> diagonal(nelem);

    // Fill the elements of the covariance matrix.

    # pragma omp parallel
    {
        std::vector <int> myrows, mycols;
        std::vector <double> myvals;

        # pragma omp for schedule(static, 10)
        for (int i = 0; i < nelem; ++i) {
            double coord[3];
            ind2coord(i + ind_start, coord);
            diagonal[i] = cov_eval(coord, coord);
        }

        # pragma omp for schedule(static, 10)
        for (int icol = 0; icol < nelem; ++icol) {
            // Translate indices into coordinates
            double colcoord[3];
            ind2coord(icol + ind_start, colcoord);
            for (int irow = icol; irow < nelem; ++irow) {
                // Evaluate the covariance between the two coordinates
                double rowcoord[3];
                ind2coord(irow + ind_start, rowcoord);
                if (fabs(colcoord[0] - rowcoord[0]) > rcorr) continue;
                if (fabs(colcoord[1] - rowcoord[1]) > rcorr) continue;
                if (fabs(colcoord[2] - rowcoord[2]) > rcorr) continue;

                double val = cov_eval(colcoord, rowcoord);
                if (icol == irow) {
                    // Regularize the matrix by promoting the diagonal
                    val *= 1.01;
                }

                // If the covariance exceeds the threshold, add it to the
                // sparse matrix
                if (val * val > 1e-6 * diagonal[icol] * diagonal[irow]) {
                    myrows.push_back(irow);
                    mycols.push_back(icol);
                    myvals.push_back(val);
                }
            }
        }
        # pragma omp critical
        {
            rows.insert(rows.end(), myrows.begin(), myrows.end());
            cols.insert(cols.end(), mycols.begin(), mycols.end());
            vals.insert(vals.end(), myvals.begin(), myvals.end());
        }
    }

    tm.stop();
    if (verbosity > 0) {
        tm.report("Sparse covariance evaluated in");
    }

    tm.start();

    // stype > 0 means that only the lower diagonal
    // elements of the symmetric matrix are needed.
    int stype = 1;
    size_t nnz = vals.size();

    cholmod_triplet * cov_triplet = cholmod_allocate_triplet(nelem,
                                                             nelem,
                                                             nnz,
                                                             stype,
                                                             CHOLMOD_REAL,
                                                             chcommon);
    memcpy(cov_triplet->i, rows.data(), nnz * sizeof(int));
    memcpy(cov_triplet->j, cols.data(), nnz * sizeof(int));
    memcpy(cov_triplet->x, vals.data(), nnz * sizeof(double));
    std::vector <int>().swap(rows); // Ensure vector is freed
    std::vector <int>().swap(cols);
    std::vector <double>().swap(vals);
    cov_triplet->nnz = nnz;

    cholmod_sparse * cov_sparse = cholmod_triplet_to_sparse(cov_triplet,
                                                            nnz,
                                                            chcommon);
    if (chcommon->status != CHOLMOD_OK) throw std::runtime_error(
                  "cholmod_triplet_to_sparse failed.");
    cholmod_free_triplet(&cov_triplet, chcommon);

    tm.stop();
    if (verbosity > 0) {
        tm.report("Sparse covariance constructed in");
    }

    // Report memory usage

    double tot_mem = (nelem * sizeof(int) + nnz * (sizeof(int) + sizeof(double)))
                     / pow(2.0, 20.0);
    double max_mem = (nelem * nelem * sizeof(double)) / pow(2.0, 20.0);
    if (verbosity > 0) {
        std::cerr << std::endl;
        std::cerr << rank << " : Allocated " << tot_mem
                  << " MB for the sparse covariance matrix. "
                  << "Compression: " << tot_mem / max_mem << std::endl;
    }

    return cov_sparse;
}

double toast::atm_sim::cov_eval(double * coord1, double * coord2) {
    // Evaluate the atmospheric absorption covariance between two coordinates
    // Church (1995) Eq.(6) & (9)
    // Coordinates are in the horizontal frame

    const long nn = 1;

    // Uncomment these lines for smoothing
    // const double ndxinv = xxstep / (nn-1);
    // const double ndzinv = zzstep / (nn-1);
    const double ninv = 1.; // / (nn * nn);

    double val = 0;

    for (int ii1 = 0; ii1 < nn; ++ii1) {
        double xx1 = coord1[0];
        double yy1 = coord1[1];
        double zz1 = coord1[2];

        // Uncomment these lines for smoothing
        // if ( ii1 ) {
        //    xx1 += ii1 * ndxinv;
        //    zz1 += ii1 * ndzinv;
        // }

        for (int ii2 = 0; ii2 < nn; ++ii2) {
            double xx2 = coord2[0];
            double yy2 = coord2[1];
            double zz2 = coord2[2];

            // Uncomment these lines for smoothing
            // if ( ii2 ) {
            //    xx2 += ii2 * ndxinv;
            //    zz2 += ii2 * ndzinv;
            // }

            double dx = xx1 - xx2;
            double dy = yy1 - yy2;
            double dz = zz1 - zz2;
            double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < rcorrsq) {
                double r = sqrt(r2);

                // Water vapor altitude factor

                double chi1 = std::exp(-(zz1 + zz2) * z0inv);

                // Kolmogorov factor

                double chi2 = kolmogorov(r);

                val += chi1 * chi2;
            }
        }
    }

    return val * ninv;
}

cholmod_sparse * toast::atm_sim::sqrt_sparse_covariance(cholmod_sparse * cov,
                                                        long ind_start,
                                                        long ind_stop) {
    /*
       Cholesky-factorize the provided sparse matrix and return the
       sparse matrix representation of the factorization
     */

    size_t nelem = ind_stop - ind_start; // Number of elements in the slice

    toast::Timer tm;
    tm.start();

    if (verbosity > 0) {
        std::cerr << rank
                  << " : Analyzing sparse covariance ... " << std::endl;
    }

    cholmod_factor * factorization;
    const int ntry = 4;
    for (int itry = 0; itry < ntry; ++itry) {
        factorization = cholmod_analyze(cov, chcommon);
        if (chcommon->status != CHOLMOD_OK) throw std::runtime_error(
                      "cholmod_analyze failed.");
        if (verbosity > 0) {
            std::cerr << rank
                      << " : Factorizing sparse covariance ... " << std::endl;
        }
        cholmod_factorize(cov, factorization, chcommon);
        if (chcommon->status != CHOLMOD_OK) {
            cholmod_free_factor(&factorization, chcommon);
            if (itry < ntry - 1) {
                // Extract band diagonal of the matrix and try
                // factorizing again
                // int ndiag = ntry - itry - 1;
                int ndiag = nelem - nelem * (itry + 1) / ntry;
                if (ndiag < 3) ndiag = 3;
                int iupper = ndiag - 1;
                int ilower = -iupper;
                if (verbosity > 0) {
                    cholmod_print_sparse(cov, "Covariance matrix", chcommon);

                    // DEBUG begin
                    if (itry > 2) {
                        FILE * covfile = fopen("failed_covmat.mtx", "w");
                        cholmod_write_sparse(covfile, cov, NULL, NULL, chcommon);
                        fclose(covfile);
                        exit(-1);
                    }

                    // DEBUG end
                    std::cerr << rank
                              << " : Factorization failed, trying a band "
                              << "diagonal matrix. ndiag = " << ndiag
                              << std::endl;
                }
                int mode = 1; // Numerical (not pattern) matrix
                cholmod_band_inplace(ilower, iupper, mode, cov, chcommon);
                if (chcommon->status != CHOLMOD_OK) throw std::runtime_error(
                              "cholmod_band_inplace failed.");
            } else throw std::runtime_error("cholmod_factorize failed.");
        } else {
            break;
        }
    }

    tm.stop();
    if (verbosity > 0) {
        tm.report("Cholesky decomposition done in");
        std::cout << " s. N = " << nelem << std::endl;
    }

    // Report memory usage (only counting the non-zero elements, no
    // supernode information)

    size_t nnz = factorization->nzmax;
    double tot_mem = (nelem * sizeof(int) + nnz * (sizeof(int) + sizeof(double)))
                     / pow(2.0, 20.0);
    if (verbosity > 0) {
        std::cerr << std::endl;
        std::cerr << rank << " : Allocated " << tot_mem
                  << " MB for the sparse factorization." << std::endl;
    }

    cholmod_sparse * sqrt_cov = cholmod_factor_to_sparse(factorization, chcommon);
    if (chcommon->status != CHOLMOD_OK) throw std::runtime_error(
                  "cholmod_factor_to_sparse failed.");
    cholmod_free_factor(&factorization, chcommon);

    // Report memory usage

    nnz = sqrt_cov->nzmax;
    tot_mem = (nelem * sizeof(int) + nnz * (sizeof(int) + sizeof(double)))
              / pow(2.0, 20.0);
    double max_mem = (nelem * nelem * sizeof(double)) / pow(2.0, 20.0);
    if (verbosity > 0) {
        std::cerr << std::endl;
        std::cerr << rank << " : Allocated " << tot_mem
                  << " MB for the sparse sqrt covariance matrix. "
                  << "Compression: " << tot_mem / max_mem << std::endl;
    }

    return sqrt_cov;
}

void toast::atm_sim::apply_sparse_covariance(cholmod_sparse * sqrt_cov,
                                             long ind_start,
                                             long ind_stop) {
    // Apply the Cholesky-decomposed (square-root) sparse covariance
    // matrix to a vector of Gaussian random numbers to impose the
    // desired correlation properties.

    toast::Timer tm;
    tm.start();

    size_t nelem = ind_stop - ind_start; // Number of elements in the slice

    // Draw the Gaussian variates in a single call

    cholmod_dense * noise_in = cholmod_allocate_dense(nelem, 1, nelem,
                                                      CHOLMOD_REAL, chcommon);
    toast::rng_dist_normal(nelem, key1, key2, counter1, counter2,
                           (double *)noise_in->x);

    cholmod_dense * noise_out = cholmod_allocate_dense(nelem, 1, nelem,
                                                       CHOLMOD_REAL, chcommon);

    // Apply the sqrt covariance to impose correlations

    int notranspose = 0;
    double one[2] = {1, 0};  // Complex one
    double zero[2] = {0, 0}; // Complex zero

    cholmod_sdmult(sqrt_cov, notranspose, one, zero, noise_in, noise_out, chcommon);
    if (chcommon->status != CHOLMOD_OK) throw std::runtime_error(
                  "cholmod_sdmult failed.");
    cholmod_free_dense(&noise_in, chcommon);

    // Subtract the mean of the slice to reduce step between the slices

    double * p = (double *)noise_out->x;
    double mean = 0, var = 0;
    for (long i = 0; i < nelem; ++i) {
        mean += p[i];
        var += p[i] * p[i];
    }
    mean /= nelem;
    var = var / nelem - mean * mean;
    for (long i = 0; i < nelem; ++i) p[i] -= mean;

    tm.stop();
    if (verbosity > 0) {
        std::ostringstream o;
        o << "Realization slice (" << ind_start << " -- "
          << ind_stop << ") var = " << var << ", constructed in";
        tm.report(o.str().c_str());
    }

    if (verbosity > 10) {
        std::ofstream f;
        std::ostringstream fname;
        fname << "realization_"
              << ind_start << "_" << ind_stop << ".txt";
        f.open(fname.str(), std::ios::out);
        for (long ielem = 0; ielem < nelem; ielem++) {
            double coord[3];
            ind2coord(ielem, coord);
            f << coord[0] << " " << coord[1] << " " << coord[2] << " "
              << p[ielem]  << std::endl;
        }
        f.close();
    }

    // Copy the slice realization over appropriate indices in
    // the full realization
    // FIXME: This is where we would blend slices

    for (long i = ind_start; i < ind_stop; ++i) {
        (*realization)[i] = p[i - ind_start];
    }

    cholmod_free_dense(&noise_out, chcommon);

    return;
}

#endif // ifdef HAVE_CHOLMOD
