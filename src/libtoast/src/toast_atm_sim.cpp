
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


void toast::atm_sim_kolmogorov_init_rank(
    int64_t nr,
    double * kolmo_x,
    double * kolmo_y,
    double rmin_kolmo,
    double rmax_kolmo,
    double rstep,
    double lmin,
    double lmax,
    int ntask,
    int rank
    ) {
    // Numerically integrate the modified Kolmogorov spectrum for the
    // correlation function at grid points. We integrate down from
    // 10*kappamax to 0 for numerical precision

    std::ostringstream o;
    o.precision(16);
    auto & logger = toast::Logger::get();

    double rstep_inv = 1. / rstep;

    std::fill(kolmo_x, kolmo_x + nr, 0);
    std::fill(kolmo_y, kolmo_y + nr, 0);

    double kappamin = 1. / lmax;
    double kappamax = 1. / lmin;
    double kappal = 0.9 * kappamax;
    double invkappal = 1 / kappal;     // Optimize
    double kappa0 = 0.75 * kappamin;
    double kappa0sq = kappa0 * kappa0; // Optimize
    int64_t nkappa = 10000;            // Number of integration steps
    double kappastart = 1e-4;
    double kappastop = 10 * kappamax;

    // kappa = exp(ikappa * kappascale) * kappastart
    double kappascale = log(kappastop / kappastart) / (nkappa - 1);

    double slope1 = 7. / 6.;
    double slope2 = -11. / 6.;

    if ((rank == 0) && atm_verbose()) {
        o.str("");
        o << std::endl;
        o << "Evaluating Kolmogorov correlation at " << nr
          << " different separations in range " << rmin_kolmo
          << " - " << rmax_kolmo << " m" << std::endl;
        o << "kappamin = " << kappamin
          << " 1/m, kappamax =  " << kappamax
          << " 1/m. nkappa = " << nkappa << std::endl;
        logger.verbose(o.str().c_str());
    }

    // Use Newton's method to integrate the correlation function

    int64_t nkappa_task = nkappa / ntask + 1;
    int64_t first_kappa = nkappa_task * rank;
    int64_t last_kappa = first_kappa + nkappa_task;
    if (last_kappa > nkappa) last_kappa = nkappa;

    // Precalculate the power spectrum function

    std::vector <double> phi(last_kappa - first_kappa);
    std::vector <double> kappa(last_kappa - first_kappa);

    # pragma omp parallel for schedule(static, 10)
    for (int64_t ikappa = first_kappa; ikappa < last_kappa; ++ikappa) {
        kappa[ikappa - first_kappa] = exp(ikappa * kappascale) * kappastart;
    }

    # pragma omp parallel for schedule(static, 10)
    for (int64_t ikappa = first_kappa; ikappa < last_kappa; ++ikappa) {
        double k = kappa[ikappa - first_kappa];
        double kkl = k * invkappal;
        phi[ikappa - first_kappa] =
            (1. + 1.802 * kkl - 0.254 * pow(kkl, slope1))
            * exp(-kkl * kkl) * pow(k * k + kappa0sq, slope2);
    }

    if (atm_verbose()) {
        std::ofstream f;
        std::ostringstream fname;
        fname << "kolmogorov_f_" << rank << ".txt";
        f.open(fname.str(), std::ios::out);
        for (int64_t ikappa = first_kappa; ikappa < last_kappa; ++ikappa) {
            f << ikappa << " " << kappa[ikappa - first_kappa] << " " <<
                phi[ikappa - first_kappa] <<
                std::endl;
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
    for (int64_t ir = 0; ir < nr; ++ir) {
        double r = rmin_kolmo
                   + (exp(ir * nri * tau) - 1) * enorm *
                   (rmax_kolmo - rmin_kolmo);
        double rinv = 1 / r;
        double val = 0;
        if (r * kappamax < 1e-2) {
            // special limit r -> 0,
            // sin(kappa.r)/r -> kappa - kappa^3*r^2/3!
            double r2 = r * r;
            for (int64_t ikappa = first_kappa; ikappa < last_kappa - 1; ++ikappa) {
                double k = kappa[ikappa - first_kappa];
                double kstep = kappa[ikappa + 1 - first_kappa] - k;
                double kappa2 = k * k;
                double kappa4 = kappa2 * kappa2;
                val += phi[ikappa - first_kappa] *
                       (kappa2 - r2 * kappa4 * ifac3) *
                       kstep;
            }
        } else {
            for (int64_t ikappa = first_kappa; ikappa < last_kappa - 1; ++ikappa) {
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

    return;
}

double toast::atm_sim_kolmogorov(
    double const & r,
    int64_t const & nr,
    double const & rmin_kolmo,
    double const & rmax_kolmo,
    double const * kolmo_x,
    double const * kolmo_y
    ) {
    // Return autocovariance of a Kolmogorov process at separation r

    if (r == 0) return kolmo_y[0];

    if (r == rmax_kolmo) return kolmo_y[nr - 1];

    if ((r < rmin_kolmo) || (r > rmax_kolmo)) {
        std::ostringstream o;
        o.precision(16);
        o << "Kolmogorov value requested at " << r
          << ", outside gridded range [" << rmin_kolmo << ", " << rmax_kolmo <<
            "].";
        auto & logger = toast::Logger::get();
        logger.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }

    // Simple linear interpolation for now.  Use a bisection method
    // to find the right elements.

    int64_t low = 0;
    int64_t high = nr - 1;
    int64_t ir;

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

double toast::atm_sim_cov_eval(
    int64_t const & nr,
    double const & rmin_kolmo,
    double const & rmax_kolmo,
    double const * kolmo_x,
    double const * kolmo_y,
    double const & rcorrsq,
    double const & z0inv,
    double * coord1,
    double * coord2,
    bool smooth,
    double xxstep,
    double zzstep
    ) {
    // Evaluate the atmospheric absorption covariance between two coordinates
    // Church (1995) Eq.(6) & (9)
    // Coordinates are in the horizontal frame

    // TK: does this make sense?
    const int64_t nn = 1;
    const double ninv = 1.; // / (nn * nn)

    double val = 0;

    if (smooth) {
        // TK: copied from commented out code in original, but if "nn" == 1, this
        // is a divide-by-zero...

        // const double ndxinv = xxstep / (nn - 1);
        // const double ndzinv = zzstep / (nn - 1);
        //
        // for (int ii1 = 0; ii1 < nn; ++ii1) {
        //     double xx1 = coord1[0];
        //     double yy1 = coord1[1];
        //     double zz1 = coord1[2];
        //
        //     if (ii1) {
        //         xx1 += ii1 * ndxinv;
        //         zz1 += ii1 * ndzinv;
        //     }
        //
        //     for (int ii2 = 0; ii2 < nn; ++ii2) {
        //         double xx2 = coord2[0];
        //         double yy2 = coord2[1];
        //         double zz2 = coord2[2];
        //
        //         if (ii2) {
        //             xx2 += ii2 * ndxinv;
        //             zz2 += ii2 * ndzinv;
        //         }
        //
        //         double dx = xx1 - xx2;
        //         double dy = yy1 - yy2;
        //         double dz = zz1 - zz2;
        //         double r2 = dx * dx + dy * dy + dz * dz;
        //
        //         if (r2 < rcorrsq) {
        //             double r = sqrt(r2);
        //
        //             // Water vapor altitude factor
        //             double chi1 = std::exp(-(zz1 + zz2) * z0inv);
        //
        //             // Kolmogorov factor
        //             double chi2 = kolmogorov(
        //                 r,
        //                 nr,
        //                 rmin_kolmo,
        //                 rmax_kolmo,
        //                 kolmo_x,
        //                 kolmo_y
        //                 );
        //
        //             val += chi1 * chi2;
        //         }
        //     }
        // }
    } else {
        for (int ii1 = 0; ii1 < nn; ++ii1) {
            double xx1 = coord1[0];
            double yy1 = coord1[1];
            double zz1 = coord1[2];

            for (int ii2 = 0; ii2 < nn; ++ii2) {
                double xx2 = coord2[0];
                double yy2 = coord2[1];
                double zz2 = coord2[2];

                double dx = xx1 - xx2;
                double dy = yy1 - yy2;
                double dz = zz1 - zz2;
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < rcorrsq) {
                    double r = sqrt(r2);

                    // Water vapor altitude factor
                    double chi1 = std::exp(-(zz1 + zz2) * z0inv);

                    // Kolmogorov factor
                    double chi2 = toast::atm_sim_kolmogorov(
                        r,
                        nr,
                        rmin_kolmo,
                        rmax_kolmo,
                        kolmo_x,
                        kolmo_y
                        );

                    val += chi1 * chi2;
                }
            }
        }
    }

    return val * ninv;
}

void toast::atm_sim_ind2coord(
    double const & xstart,
    double const & ystart,
    double const & zstart,
    double const & xstep,
    double const & ystep,
    double const & zstep,
    int64_t const & xstride,
    int64_t const & ystride,
    int64_t const & zstride,
    double const & xstrideinv,
    double const & ystrideinv,
    double const & zstrideinv,
    double const & cosel0,
    double const & sinel0,
    int64_t const * full_index,
    int64_t const & i,
    double * coord
    ) {
    // Translate a compressed index into xyz-coordinates
    // in the horizontal frame

    int64_t ifull = full_index[i];

    // TK: these are mixed types being implicitly cast back to int64.
    int64_t ix = ifull * xstrideinv;
    int64_t iy = (ifull - ix * xstride) * ystrideinv;
    int64_t iz = ifull - ix * xstride - iy * ystride;

    // coordinates in the scan frame

    double x = xstart + ix * xstep;
    double y = ystart + iy * ystep;
    double z = zstart + iz * zstep;

    // Into the horizontal frame

    coord[0] = x * cosel0 - z * sinel0;
    coord[1] = y;
    coord[2] = x * sinel0 + z * cosel0;
}

int64_t toast::atm_sim_coord2ind(
    double const & xstart,
    double const & ystart,
    double const & zstart,
    int64_t const & xstride,
    int64_t const & ystride,
    int64_t const & zstride,
    double const & xstepinv,
    double const & ystepinv,
    double const & zstepinv,
    int64_t const & nx,
    int64_t const & ny,
    int64_t const & nz,
    int64_t const * compressed_index,
    double const & x,
    double const & y,
    double const & z
    ) {
    // Translate scan frame xyz-coordinates into a compressed index

    // TK: these are mixed types being implicitly cast back to int64.
    int64_t ix = (x - xstart) * xstepinv;
    int64_t iy = (y - ystart) * ystepinv;
    int64_t iz = (z - zstart) * zstepinv;

# ifndef NO_ATM_CHECKS
    if ((ix < 0) || (ix > nx - 1) || (iy < 0) || (iy > ny - 1) || (iz < 0) ||
        (iz > nz - 1)) {
        std::ostringstream o;
        o.precision(16);
        o << "atmsim::coord2ind : full index out of bounds at ("
          << x << ", " << y << ", " << z << ") = ("
          << ix << " /  " << nx << ", " << iy << " / " << ny << ", "
          << iz << ", " << nz << ")";
        auto & logger = toast::Logger::get();
        logger.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
# endif // ifndef NO_ATM_CHECKS

    size_t ifull = ix * xstride + iy * ystride + iz * zstride;

    return compressed_index[ifull];
}

cholmod_sparse * toast::atm_sim_build_sparse_covariance(
    int64_t ind_start,
    int64_t ind_stop,
    int64_t nr,
    double rmin_kolmo,
    double rmax_kolmo,
    double const * kolmo_x,
    double const * kolmo_y,
    double rcorr,
    double xstart,
    double ystart,
    double zstart,
    double xstep,
    double ystep,
    double zstep,
    int64_t xstride,
    int64_t ystride,
    int64_t zstride,
    double z0,
    double cosel0,
    double sinel0,
    int64_t const * full_index,
    bool smooth,
    double xxstep,
    double zzstep,
    int rank
    ) {
    // Build a sparse covariance matrix.

    auto & chol = toast::CholmodCommon::get();
    auto & logger = Logger::get();
    std::ostringstream o;

    toast::Timer timer;
    timer.start();

    double xstrideinv = 1.0 / xstride;
    double ystrideinv = 1.0 / ystride;
    double zstrideinv = 1.0 / zstride;

    double rcorrsq = rcorr * rcorr;

    double z0inv = 1. / (2. * z0);

    // Build the covariance matrix first in the triplet form, then
    // cast it to the column-packed format.

    std::vector <int> rows;
    std::vector <int> cols;
    std::vector <double> vals;
    int64_t nelem = ind_stop - ind_start; // Number of elements in the slice
    std::vector <double> diagonal(nelem);

    // Fill the elements of the covariance matrix.

    # pragma omp parallel
    {
        std::vector <int> myrows, mycols;
        std::vector <double> myvals;

        # pragma omp for schedule(static, 10)
        for (int64_t i = 0; i < nelem; ++i) {
            double coord[3];
            toast::atm_sim_ind2coord(
                xstart,
                ystart,
                zstart,
                xstep,
                ystep,
                zstep,
                xstride,
                ystride,
                zstride,
                xstrideinv,
                ystrideinv,
                zstrideinv,
                cosel0,
                sinel0,
                full_index,
                i + ind_start,
                coord
                );
            diagonal[i] = toast::atm_sim_cov_eval(
                nr,
                rmin_kolmo,
                rmax_kolmo,
                kolmo_x,
                kolmo_y,
                rcorrsq,
                z0inv,
                coord,
                coord,
                smooth,
                xxstep,
                zzstep
                );
        }

        # pragma omp for schedule(static, 10)
        for (int64_t icol = 0; icol < nelem; ++icol) {
            // Translate indices into coordinates
            double colcoord[3];
            toast::atm_sim_ind2coord(
                xstart,
                ystart,
                zstart,
                xstep,
                ystep,
                zstep,
                xstride,
                ystride,
                zstride,
                xstrideinv,
                ystrideinv,
                zstrideinv,
                cosel0,
                sinel0,
                full_index,
                icol + ind_start,
                colcoord
                );
            for (int64_t irow = icol; irow < nelem; ++irow) {
                // Evaluate the covariance between the two coordinates
                double rowcoord[3];
                toast::atm_sim_ind2coord(
                    xstart,
                    ystart,
                    zstart,
                    xstep,
                    ystep,
                    zstep,
                    xstride,
                    ystride,
                    zstride,
                    xstrideinv,
                    ystrideinv,
                    zstrideinv,
                    cosel0,
                    sinel0,
                    full_index,
                    irow + ind_start,
                    rowcoord
                    );
                if (fabs(colcoord[0] - rowcoord[0]) > rcorr) continue;
                if (fabs(colcoord[1] - rowcoord[1]) > rcorr) continue;
                if (fabs(colcoord[2] - rowcoord[2]) > rcorr) continue;

                double val = toast::atm_sim_cov_eval(
                    nr,
                    rmin_kolmo,
                    rmax_kolmo,
                    kolmo_x,
                    kolmo_y,
                    rcorrsq,
                    z0inv,
                    colcoord,
                    rowcoord,
                    smooth,
                    xxstep,
                    zzstep
                    );
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

    timer.stop();

    if (atm_verbose()) {
        o.str("");
        o << rank << " : Sparse covariance evaluated in "
          << timer.seconds() << " s.";
        logger.verbose(o.str().c_str());
    }

    timer.start();

    // stype > 0 means that only the lower diagonal
    // elements of the symmetric matrix are needed.
    int stype = 1;
    size_t nnz = vals.size();

    cholmod_triplet * cov_triplet = cholmod_allocate_triplet(nelem,
                                                             nelem,
                                                             nnz,
                                                             stype,
                                                             CHOLMOD_REAL,
                                                             chol.chcommon);
    memcpy(cov_triplet->i, rows.data(), nnz * sizeof(int));
    memcpy(cov_triplet->j, cols.data(), nnz * sizeof(int));
    memcpy(cov_triplet->x, vals.data(), nnz * sizeof(double));
    std::vector <int>().swap(rows); // Ensure vector is freed
    std::vector <int>().swap(cols);
    std::vector <double>().swap(vals);
    cov_triplet->nnz = nnz;

    cholmod_sparse * cov_sparse = cholmod_triplet_to_sparse(cov_triplet,
                                                            nnz,
                                                            chol.chcommon);
    if (chol.chcommon->status != CHOLMOD_OK) {
        throw std::runtime_error("cholmod_triplet_to_sparse failed.");
    }

    cholmod_free_triplet(&cov_triplet, chol.chcommon);

    timer.stop();

    if (atm_verbose()) {
        o.str("");
        o << rank << " : Sparse covariance constructed in "
          << timer.seconds() << " s.";
        logger.verbose(o.str().c_str());
    }

    // Report memory usage

    double tot_mem =
        (nelem * sizeof(int) + nnz * (sizeof(int) + sizeof(double)))
        / pow(2.0, 20.0);

    double max_mem = (nelem * nelem * sizeof(double)) / pow(2.0, 20.0);
    if (atm_verbose()) {
        o.str("");
        o << rank << " : Allocated " << tot_mem
          << " MB for the sparse covariance matrix. "
          << "Compression: " << tot_mem / max_mem;
        logger.verbose(o.str().c_str());
    }

    return cov_sparse;
}

cholmod_sparse * toast::atm_sim_sqrt_sparse_covariance(
    cholmod_sparse * cov,
    int64_t ind_start,
    int64_t ind_stop,
    int rank
    ) {
    // Cholesky-factorize the provided sparse matrix and return the
    // sparse matrix representation of the factorization

    auto & chol = toast::CholmodCommon::get();
    auto & logger = Logger::get();
    std::ostringstream o;

    int64_t nelem = ind_stop - ind_start; // Number of elements in the slice

    toast::Timer timer;
    timer.start();

    if (atm_verbose()) {
        o.str("");
        o << rank << " : Analyzing sparse covariance ... ";
        logger.verbose(o.str().c_str());
    }

    cholmod_factor * factorization;

    const int ntry = 4;

    for (int itry = 0; itry < ntry; ++itry) {
        factorization = cholmod_analyze(cov, chol.chcommon);

        if (chol.chcommon->status != CHOLMOD_OK) {
            throw std::runtime_error("cholmod_analyze failed.");
        }

        if (atm_verbose()) {
            o.str("");
            o << rank << " : Factorizing sparse covariance ... ";
            logger.verbose(o.str().c_str());
        }

        cholmod_factorize(cov, factorization, chol.chcommon);

        if (chol.chcommon->status != CHOLMOD_OK) {
            cholmod_free_factor(&factorization, chol.chcommon);
            if (itry < ntry - 1) {
                // Extract band diagonal of the matrix and try
                // factorizing again
                // int ndiag = ntry - itry - 1;
                int ndiag = nelem - nelem * (itry + 1) / ntry;
                if (ndiag < 3) ndiag = 3;
                int iupper = ndiag - 1;
                int ilower = -iupper;
                if (atm_verbose()) {
                    cholmod_print_sparse(cov, "Covariance matrix", chol.chcommon);

                    // DEBUG begin
                    if (itry > 2) {
                        FILE * covfile = fopen("failed_covmat.mtx", "w");
                        cholmod_write_sparse(covfile, cov, NULL, NULL,
                                             chol.chcommon);
                        fclose(covfile);
                        exit(-1);
                    }

                    // DEBUG end
                    o.str("");
                    o << rank
                      << " : Factorization failed, trying a band "
                      << "diagonal matrix. ndiag = " << ndiag;
                    logger.warning(o.str().c_str());
                }
                int mode = 1; // Numerical (not pattern) matrix
                cholmod_band_inplace(ilower, iupper, mode, cov, chol.chcommon);

                if (chol.chcommon->status != CHOLMOD_OK) {
                    throw std::runtime_error("cholmod_band_inplace failed.");
                }
            } else throw std::runtime_error("cholmod_factorize failed.");
        } else {
            break;
        }
    }

    timer.stop();

    if (atm_verbose()) {
        o.str("");
        o << rank
          << " : Cholesky decomposition done in " << timer.seconds()
          << " s. N = " << nelem << std::endl;
        logger.verbose(o.str().c_str());
    }

    // Report memory usage (only counting the non-zero elements, no
    // supernode information)

    size_t nnz = factorization->nzmax;

    double tot_mem =
        (nelem * sizeof(int) + nnz * (sizeof(int) + sizeof(double)))
        / pow(2.0, 20.0);

    if (atm_verbose()) {
        o.str("");
        o << rank << " : Allocated " << tot_mem
          << " MB for the sparse factorization.";
        logger.verbose(o.str().c_str());
    }

    cholmod_sparse * sqrt_cov = cholmod_factor_to_sparse(factorization, chol.chcommon);

    if (chol.chcommon->status != CHOLMOD_OK) {
        throw std::runtime_error("cholmod_factor_to_sparse failed.");
    }

    cholmod_free_factor(&factorization, chol.chcommon);

    // Report memory usage

    nnz = sqrt_cov->nzmax;

    tot_mem = (nelem * sizeof(int) + nnz * (sizeof(int) + sizeof(double)))
              / pow(2.0, 20.0);

    double max_mem = (nelem * nelem * sizeof(double)) / pow(2.0, 20.0);

    if (atm_verbose()) {
        o.str("");
        o << rank << " : Allocated " << tot_mem
          << " MB for the sparse sqrt covariance matrix. "
          << "Compression: " << tot_mem / max_mem;
        logger.verbose(o.str().c_str());
    }

    return sqrt_cov;
}

void toast::atm_sim_apply_sparse_covariance(
    cholmod_sparse * sqrt_cov,
    int64_t ind_start,
    int64_t ind_stop,
    uint64_t key1,
    uint64_t key2,
    uint64_t counter1,
    uint64_t counter2,
    double * realization,
    int rank
    ) {
    // Apply the Cholesky-decomposed (square-root) sparse covariance
    // matrix to a vector of Gaussian random numbers to impose the
    // desired correlation properties.

    auto & chol = toast::CholmodCommon::get();
    auto & logger = Logger::get();
    std::ostringstream o;

    toast::Timer timer;
    timer.start();

    int64_t nelem = ind_stop - ind_start; // Number of elements in the slice

    // Draw the Gaussian variates in a single call

    cholmod_dense * noise_in = cholmod_allocate_dense(nelem, 1, nelem,
                                                      CHOLMOD_REAL, chol.chcommon);

    toast::rng_dist_normal(nelem, key1, key2, counter1, counter2,
                           (double *)noise_in->x);

    cholmod_dense * noise_out = cholmod_allocate_dense(nelem, 1, nelem,
                                                       CHOLMOD_REAL, chol.chcommon);

    // Apply the sqrt covariance to impose correlations

    int notranspose = 0;
    double one[2] = {1, 0};  // Complex one
    double zero[2] = {0, 0}; // Complex zero

    cholmod_sdmult(sqrt_cov, notranspose, one, zero, noise_in, noise_out,
                   chol.chcommon);

    if (chol.chcommon->status != CHOLMOD_OK) {
        throw std::runtime_error("cholmod_sdmult failed.");
    }

    cholmod_free_dense(&noise_in, chol.chcommon);

    // Subtract the mean of the slice to reduce step between the slices

    double * p = (double *)noise_out->x;
    double mean = 0;
    double var = 0;
    for (int64_t i = 0; i < nelem; ++i) {
        mean += p[i];
        var += p[i] * p[i];
    }
    mean /= nelem;
    var = var / nelem - mean * mean;
    for (int64_t i = 0; i < nelem; ++i) p[i] -= mean;

    timer.stop();

    if (atm_verbose()) {
        o.str("");
        o << rank
          << " : Realization slice (" << ind_start << " -- " << ind_stop
          << ") var = " << var << ", constructed in "
          << timer.seconds() << " s.";
        logger.verbose(o.str().c_str());
    }

    // Copy the slice realization over appropriate indices in
    // the full realization
    // FIXME: This is where we would blend slices

    for (int64_t i = ind_start; i < ind_stop; ++i) {
        realization[i] = p[i - ind_start];
    }

    cholmod_free_dense(&noise_out, chol.chcommon);

    return;
}

void toast::atm_sim_compute_slice(
    int64_t ind_start,
    int64_t ind_stop,
    int64_t nr,
    double rmin_kolmo,
    double rmax_kolmo,
    double const * kolmo_x,
    double const * kolmo_y,
    double rcorr,
    double xstart,
    double ystart,
    double zstart,
    double xstep,
    double ystep,
    double zstep,
    int64_t xstride,
    int64_t ystride,
    int64_t zstride,
    double z0,
    double cosel0,
    double sinel0,
    int64_t const * full_index,
    bool smooth,
    double xxstep,
    double zzstep,
    int rank,
    uint64_t key1,
    uint64_t key2,
    uint64_t counter1,
    uint64_t counter2,
    double * realization
    ) {
    auto & chol = toast::CholmodCommon::get();

    auto & gt = toast::GlobalTimers::get();

    gt.start("atm_sim_build_sparse_covariance");

    cholmod_sparse * cov = toast::atm_sim_build_sparse_covariance(
        ind_start,
        ind_stop,
        nr,
        rmin_kolmo,
        rmax_kolmo,
        kolmo_x,
        kolmo_y,
        rcorr,
        xstart,
        ystart,
        zstart,
        xstep,
        ystep,
        zstep,
        xstride,
        ystride,
        zstride,
        z0,
        cosel0,
        sinel0,
        full_index,
        smooth,
        xxstep,
        zzstep,
        rank
        );

    gt.stop("atm_sim_build_sparse_covariance");

    gt.start("atm_sim_sqrt_sparse_covariance");

    cholmod_sparse * sqrt_cov = toast::atm_sim_sqrt_sparse_covariance(
        cov,
        ind_start,
        ind_stop,
        rank
        );

    gt.stop("atm_sim_sqrt_sparse_covariance");

    cholmod_free_sparse(&cov, chol.chcommon);

    gt.start("atm_sim_apply_sparse_covariance");

    toast::atm_sim_apply_sparse_covariance(
        sqrt_cov,
        ind_start,
        ind_stop,
        key1,
        key2,
        counter1,
        counter2,
        realization,
        rank
        );

    gt.stop("atm_sim_apply_sparse_covariance");

    cholmod_free_sparse(&sqrt_cov, chol.chcommon);

    return;
}

#endif // ifdef HAVE_CHOLMOD
