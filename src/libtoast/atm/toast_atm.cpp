/*
  Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
  All rights reserved.  Use of this source code is governed by
  a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_atm_internal.hpp>

#include <sstream>
#include <iostream>
#include <fstream>
#include <cstring>
#include <random>
#include <functional>
#include <cmath>
#include <omp.h>


double median( std::vector<double> vec ) {
    sort( vec.begin(), vec.end());
    int half1 = (vec.size() - 1) * .5;
    int half2 = vec.size() * .5;

    return .5 *( vec[half1] + vec[half2] );
} // median


double mean( std::vector<double> vec ) {
    if ( vec.size() == 0 ) return 0;

    double sum = 0;
    for ( auto& val : vec ) sum += val;

    return sum / vec.size();
} // mean


toast::atm::sim::sim( double azmin, double azmax, double elmin, double elmax,
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
		      int verbosity, MPI_Comm comm, int gangsize, double fnear,
		      uint64_t key1,uint64_t key2,
		      uint64_t counter1, uint64_t counter2
    ) : azmin(azmin), azmax(azmax),
        elmin(elmin), elmax(elmax), tmin(tmin), tmax(tmax),
        lmin_center(lmin_center), lmin_sigma(lmin_sigma),
        lmax_center(lmax_center), lmax_sigma(lmax_sigma),
        w_center(w_center), w_sigma(w_sigma),
        wdir_center(wdir_center), wdir_sigma(wdir_sigma),
        z0_center(z0_center), z0_sigma(z0_sigma),
        T0_center(T0_center), T0_sigma(T0_sigma),
        zatm(zatm), zmax(zmax), xstep(xstep), ystep(ystep),
        zstep(zstep), nelem_sim_max(nelem_sim_max),
        verbosity(verbosity),
        comm(comm), gangsize(gangsize), fnear(fnear),
        key1(key1), key2(key2),
        counter1(counter1), counter2(counter2) {

    int ierr;
    ierr = MPI_Comm_size( comm, &ntask );
    if ( ierr != MPI_SUCCESS )
        throw std::runtime_error( "Failed to get size of MPI communicator." );
    ierr = MPI_Comm_rank( comm, &rank );
    if ( ierr != MPI_SUCCESS )
        throw std::runtime_error( "Failed to get rank in MPI communicator." );

    if ( gangsize == 1 ) {
        ngang = ntask;
        gang = rank;
        comm_gang = MPI_COMM_SELF;
        ntask_gang = 1;
        rank_gang = 0;
    } else if ( gangsize > 0 and 2*gangsize <= ntask ) {
        ngang = ntask / gangsize;
        gang = rank / gangsize;
        // If the last gang is smaller than the rest, it will be merged with
        // the second-to-last gang
        if ( gang > ngang-1 ) gang = ngang - 1;
        ierr = MPI_Comm_split( comm, gang, rank, &comm_gang );
        if ( ierr != MPI_SUCCESS )
            throw std::runtime_error( "Failed to split MPI communicator." );
        ierr = MPI_Comm_size( comm_gang, &ntask_gang );
        if ( ierr != MPI_SUCCESS )
            throw std::runtime_error( "Failed to get size of the split MPI "
                                      "communicator." );
        ierr = MPI_Comm_rank( comm_gang, &rank_gang );
        if ( ierr != MPI_SUCCESS )
            throw std::runtime_error( "Failed to get rank in the split MPI "
                                      "communicator." );
    } else {
        ngang = 1;
        gang = 0;
        comm_gang = comm;
        ntask_gang = ntask;
        rank_gang = rank;
    }

    nthread = omp_get_max_threads();

    if (rank == 0 && verbosity > 0)
        std::cerr<<"atmsim constructed with " << ntask << " processes, "
                 << ngang << " gangs, " << nthread << " threads per process."
                 << std::endl;

    if ( azmin >= azmax ) throw std::runtime_error( "atmsim: azmin >= azmax." );
    if ( elmin < 0 ) throw std::runtime_error( "atmsim: elmin < 0." );
    if ( elmax > M_PI_2 ) throw std::runtime_error( "atmsim: elmax > pi/2." );
    if ( elmin > elmax ) throw std::runtime_error( "atmsim: elmin > elmax." );
    if ( tmin > tmax ) throw std::runtime_error( "atmsim: tmin > tmax." );
    if ( lmin_center > lmax_center )
        throw std::runtime_error( "atmsim: lmin_center > lmax_center." );

    xstepinv = 1 / xstep;
    ystepinv = 1 / ystep;
    zstepinv = 1 / zstep;

    delta_az = azmax - azmin;
    delta_el = elmax - elmin;
    delta_t = tmax - tmin;

    double tol = 0.5 * M_PI / 180; // 0.5 degree tolerance
    tanmin = tan( elmin - tol ); // speed up the in-cone calculation
    tanmax = tan( elmax + tol ); // speed up the in-cone calculation
    sinmin = sin( -0.5*delta_az - tol ); // speed up the in-cone calculation
    sinmax = sin(  0.5*delta_az + tol ); // speed up the in-cone calculation

    az0 = azmin + delta_az / 2;

    fnearinv = 1 / fnear;

    if ( rank == 0 && verbosity > 0 ) {
        std::cerr << std::endl;
        std::cerr << "Input parameters:" << std::endl;
        std::cerr << "             az = [" << azmin << " - " << azmax
                  << "] (" << delta_az << " radians)" << std::endl;
        std::cerr << "             el = [" << elmin << " - " << elmax
                  << "] (" << delta_el << " radians)" << std::endl;
        std::cerr << "              t = [" << tmin << " - " << tmax
                  << "] (" << delta_t << " s)" << std::endl;
        std::cerr << "           lmin = " << lmin_center << " +- " << lmin_sigma
                  << " m" << std::endl;
        std::cerr << "           lmax = " << lmax_center << " +- " << lmax_sigma
                  << " m" << std::endl;
        std::cerr << "              w = " << w_center << " +- " << w_sigma
                  << " m" << std::endl;
        std::cerr << "           wdir = " << wdir_center << " +- " << wdir_sigma
                  << " radians " << std::endl;
        std::cerr << "             z0 = " << z0_center << " +- " << z0_sigma
                  << " m" << std::endl;
        std::cerr << "             T0 = " << T0_center << " +- " << T0_sigma
                  << " K" << std::endl;
        std::cerr << "           zatm = " << zatm << " m" << std::endl;
        std::cerr << "           zmax = " << zmax << " m" << std::endl;
        std::cerr << "          xstep = " << xstep << " m" << std::endl;
        std::cerr << "          ystep = " << ystep << " m" << std::endl;
        std::cerr << "          zstep = " << zstep << " m" << std::endl;
        std::cerr << "  nelem_sim_max = " << nelem_sim_max << std::endl;
        std::cerr << "          fnear = " << fnear << std::endl;
        std::cerr << "      verbosity = " << verbosity << std::endl;
    }

    // Initialize Elemental.  This should already be done by the top-level
    // toast::init() function, which is called when importing the toast.mpi
    // python package.

    if ( !El::Initialized() ) El::Initialize();

    // Initialize Elemental grid for distributed matrix manipulation
    grid = new El::Grid( comm_gang );
}


toast::atm::sim::~sim() {
    if ( grid ) delete grid;
}


void toast::atm::sim::simulate( bool save_covmat ) {

    try {

        draw();

        get_volume();

        compress_volume();

        if ( rank == 0 and verbosity > 0 ) {
            std::cerr << "Resizing realizations to " << nelem << std::endl;
        }

        realization.resize(nelem);
        realization_near.resize(nelem);
        realization_verynear.resize(nelem);

        double t1 = MPI_Wtime();

        long ind_start = 0, ind_stop = 0, slice = 0;

        // Simulate the atmosphere in indepedent slices, each slice
        // assigned to exactly one gang

        std::vector<int> slice_starts;
        std::vector<int> slice_stops;

        while (true) {
            get_slice(ind_start, ind_stop);
            slice_starts.push_back(ind_start);
            slice_stops.push_back(ind_stop);

            if ( slice % ngang == gang ) {
                for (int near=0; near<3; ++near) {

                    std::vector<double> *preal;
                    double scale;

                    switch ( near ) {
                    case 0 :
                        preal = &realization;
                        scale = 1;
                        break;
                    case 1 :
                        preal = &realization_near;
                        scale = fnear;
                        break;
                    case 2 :
                        preal = &realization_verynear;
                        scale = fnear * fnear;
                        break;
                    default : throw std::runtime_error( "Unknown field." );
                        break;
                    }

                    El::DistMatrix<double> *cov = build_covariance( ind_start, ind_stop,
                                                                    save_covmat, scale );
                    sqrt_covariance( cov, ind_start, ind_stop, save_covmat, near );
                    apply_covariance( cov, *preal, ind_start, ind_stop, near );

                    delete cov;
                }
            }

            if ( ind_stop == nelem ) break;

            ++slice;
        }

        // Gather the slices from the gangs

        for ( size_t slice=0; slice < slice_starts.size(); ++slice ) {
            ind_start = slice_starts[slice];
            ind_stop = slice_stops[slice];
            int root_gang = slice % ngang;
            int root = root_gang * gangsize;
            int ierr = MPI_Bcast( realization.data()+ind_start, ind_stop-ind_start,
                                  MPI_DOUBLE, root, comm );
            if ( ierr != MPI_SUCCESS )
                throw std::runtime_error( "Failed to broadcast the realization" );
            ierr = MPI_Bcast( realization_near.data()+ind_start, ind_stop-ind_start,
                              MPI_DOUBLE, root, comm );
            if ( ierr != MPI_SUCCESS )
                throw std::runtime_error( "Failed to broadcast the near realization" );
            ierr = MPI_Bcast( realization_verynear.data()+ind_start,
                              ind_stop-ind_start, MPI_DOUBLE, root, comm );
            if ( ierr != MPI_SUCCESS )
                throw std::runtime_error( "Failed to broadcast the near realization" );
        }

        //smooth();

        MPI_Barrier( comm );
        double t2 = MPI_Wtime();

        if ( rank == 0 && verbosity > 0 ) {
            std::cerr << std::endl;
            std::cerr << "Realization constructed in " << t2-t1 << " s." << std::endl;
        }

    } catch ( const std::exception& e ) {
        std::cerr << "WARNING: atm::simulate failed with: " << e.what() << std::endl;
    }

    return;
}


void toast::atm::sim::get_slice( long &ind_start, long &ind_stop ) {

    // Identify a manageable slice of compressed indices to simulate next

    ind_start = ind_stop;

    long ix_start = full_index[ind_start] / xstride;
    long ix1 = ix_start;
    long ix2;

    while ( true ) {
        ix2 = ix1;
        while ( ix1 == ix2 ) {
            ++ind_stop;
            if ( ind_stop == nelem ) break;
            ix2 = full_index[ind_stop] / xstride;
        }
        if ( ind_stop == nelem ) break;
        if ( ind_stop - ind_start > nelem_sim_max ) break;
        ix1 = ix2;
    }

    if ( rank == 0 && verbosity > 0 ) {
        std::cerr << "X-slice: " << ix_start*xstep << " -- " << ix2*xstep
                  << " ( " << nx*xstep << " )"
                  << "( " << ind_start << " -- "<< ind_stop << " ) "
                  << "( " << nelem << " )"<< std::endl;
    }

    return;
}


void toast::atm::sim::smooth() {

    // Replace each vertex with a mean of its immediate vicinity

    double t1 = MPI_Wtime();

    double coord[3];

    std::vector<double> smoothed_realization(realization.size());
    std::vector<double> smoothed_realization_near(realization.size());
    std::vector<double> smoothed_realization_verynear(realization.size());

    for ( size_t i=0; i < full_index.size(); ++i ) {
        ind2coord( i, coord );
        long ix = coord[0] * xstepinv;
        long iy = coord[1] * ystepinv;
        long iz = coord[2] * zstepinv;

        long offset = ix * xstride + iy * ystride + iz * zstride;

        long w = 3; // width of the smoothing kernel
        long ifullmax = compressed_index.size();

        std::vector<double> vals;
        std::vector<double> vals_near;
        std::vector<double> vals_verynear;

        //for (int xoff=-w; xoff <= w; ++xoff) {
        for (int xoff=0; xoff <= 0; ++xoff) {
            if ( ix + xoff < 0 ) continue;
            if ( ix + xoff >= nx ) break;

            for (int yoff=-w; yoff <= w; ++yoff) {
                if ( iy + yoff < 0 ) continue;
                if ( iy + yoff >= ny ) break;

                for (int zoff=-w; zoff <= w; ++zoff) {
                    if ( iz + zoff < 0 ) continue;
                    if ( iz + zoff >= nz ) break;

                    long ifull = offset + xoff*xstride + yoff*ystride + zoff*zstride;

                    if ( ifull < 0 || ifull >= ifullmax )
                        throw std::runtime_error( "Index out of range in smoothing." );

                    long ii = compressed_index[ifull];

                    if (ii >= 0) {
                        vals.push_back( realization[ii] );
                        vals_near.push_back( realization_near[ii] );
                        vals_verynear.push_back( realization_verynear[ii] );
                    }
                }
            }
        }

        // Get the smoothed value

        smoothed_realization[i] = mean( vals );
        smoothed_realization_near[i] = mean( vals_near );
        smoothed_realization_verynear[i] = mean( vals_verynear );
    }

    realization = smoothed_realization;
    realization_near = smoothed_realization_near;
    realization_verynear = smoothed_realization_verynear;

    double t2 = MPI_Wtime();

    if ( rank == 0 && verbosity > 0 )
        std::cerr << "Realization smoothed in " << t2-t1 << " s." << std::endl;

    return;
}


void toast::atm::sim::observe( double *t, double *az, double *el, double *tod,
			       long nsamp, double fixed_r ) {

    try {

        double t1 = MPI_Wtime();

        double rstep = xstep;
        if ( ystep < rstep ) rstep = ystep;
        if ( zstep < rstep ) rstep = zstep;
        rstep /= 3;
        //if ( fixed_r > 0 ) rstep = 1;

        // For each sample, integrate along the line of sight by summing
        // the atmosphere values. See Church (1995) Section 2.2, first equation.
        // We omit the optical depth factor which is close to unity.

        double zatm_inv = 1. / zatm;

#pragma omp parallel for schedule(static, 100)
        for ( long i=0; i<nsamp; ++i ) {

            if ( az[i] < azmin || az[i] > azmax || el[i] < elmin || el[i] > elmax ) {
                std::ostringstream o;
                o.precision( 16 );
                o << "atmsim::observe : observation out of bounds (az, el, t) = ("
                  << az[i] << ",  " << el[i] << ", " << t[i] << ") allowed: ("
                  << azmin << " - "<< azmax << ", "
                  << elmin << " - "<< elmax << ", "
                  << tmin << " - "<< tmax << ")"
                  << std::endl;
                throw std::runtime_error( o.str().c_str() );
            }

            double t_now = t[i] - tmin;
            double az_now = az[i] - az0; // Relative to center of field
            double el_now = el[i];

            double xtel_now = xtel - wx*t_now;
            double ytel_now = ytel - wy*t_now;

            double sin_el = sin( el_now );
            double cos_el = cos( el_now );
            double sin_az = sin( az_now );
            double cos_az = cos( az_now );

            double val = 0;
            double r = 1; // Start integration at a reasonable distance
            if ( fixed_r > 0 ) r = fixed_r;

            std::vector<long> last_ind(3);
            std::vector<double> last_nodes(8);

            while ( true ) {

                // Coordinates at distance r. The scan is centered on the X-axis

                int near=0;
                std::vector<double> *preal = &realization;
                double r_eff = r;
                if ( r < rverynear ) {
                    // Use the very near field simulation
                    near = 2;
                    preal = &realization_verynear;
                    r_eff *= fnearinv * fnearinv;
                } else if ( r < rnear ) {
                    // Use the near field simulation
                    near = 1;
                    preal = &realization_near;
                    r_eff *= fnearinv;
                }

                double z = r_eff * sin_el;
                if ( z >= zmax ) break;
                double rproj = r_eff * cos_el;
                double x = xtel_now + rproj * cos_az;
                double y = ytel_now - rproj * sin_az;

                double x_eff = x;
                double y_eff = y;
                double z_eff = z;
                for (int j=0; j < near; ++j) {
                    x_eff *= fnear;
                    y_eff *= fnear;
                    z_eff *= fnear;
                }

#ifdef DEBUG
                if ( x < 0 || x > delta_x ||
                     y < 0 || y > delta_y ||
                     z < 0 || z > delta_z ) {
                    std::ostringstream o;
                    o.precision( 16 );
                    o << "atmsim::observe : observation point out of bounds ("
                      << x << " / " << delta_x << ", "
                      << y << " / " << delta_y << ", "
                      << z << " / " << delta_z << ")" << std::endl
                      << "( t, t-tmin, az, az-az0, el, r, r_eff, r_proj ) = " << std::endl
                      << "( " << t[i] << ", " << t_now << ", " << az[i]
                      << ", " << az_now << ", " << el[i] << ", "
                      << r << ", " << r_eff << ", " << rproj
                      << ")" << std::endl
                      << "(x_tel, y_tel, x_tel_now, y_tel_now, wx, wy) = ("
                      << xtel << ", " << ytel << ", "
                      << xtel_now << ", " << ytel_now << ", "
                      << wx << ", " << wy << ")" << std::endl;
                    std::cerr << o.str();
                    throw std::runtime_error( o.str().c_str() );
                }
#endif

                // Combine atmospheric emission (via interpolation) with the
                // ambient temperature

                double step_val;
                try {
                    step_val = interp( *preal, x, y, z, last_ind, last_nodes )
                        * (1 - z_eff * zatm_inv);
                } catch ( const std::runtime_error& e ) {
                    std::ostringstream o;
                    o.precision( 16 );
                    o << "atmsim::observe : interp failed at ("
                      << x << " /  " << delta_x << ", " << y << " / " << delta_y << ", "
                      << z << " / " << delta_z << ")"
                      << "( t, az, el ) " << "( " << t[i] << ", " << az[i] << ", "
                      << el[i] << ") with " << e.what() << std::endl;
                    throw std::runtime_error( o.str().c_str() );
                }

                // In the near field the steps are shorter and so the weights are smaller
                for (int j=0; j < near; ++j) step_val *= fnear;
                val += step_val;

                // Prepare for the next step

                double step = rstep;
                for (int j=0; j < near; ++j) step *= fnear;

                r += step;

                if ( fixed_r > 0 ) break;
                //if ( fixed_r > 0 and r > fixed_r ) break;
            }

            tod[i] = val * rstep * T0;
        }

        double t2 = MPI_Wtime();

        if ( rank == 0 && verbosity > 0 ) {
            if ( fixed_r > 0 )
                std::cerr << nsamp << " samples observed at r =  " << fixed_r
                          << " in " << t2-t1 << " s." << std::endl;
            else
                std::cerr << nsamp << " samples observed in " << t2-t1 << " s."
                          << std::endl;
        }

    } catch ( const std::exception& e ) {
        std::cerr << "WARNING: atm::observe failed with: " << e.what() << std::endl;
    } catch ( ... ) {
        std::cerr << "WARNING: atm::observe failed with an unknown exception."
                  << std::endl;
    }

    return;
}


void toast::atm::sim::draw() {

    // Draw 100 gaussian variates to use in drawing the simulation parameters

    const size_t nrand = 100;
    double randn[nrand];
    rng::dist_normal( nrand, key1, key2, counter1, counter2, randn );
    counter2 += nrand;
    double *prand=randn;

    if ( rank == 0 ) {
        lmin = 0;
        lmax = 0;
        w = -1;
        wdir = 0;
        z0 = 0;
        T0 = 0;

        while( lmin >= lmax ){
            lmin = 0;
            lmax = 0;
            while (lmin <= 0) lmin = lmin_center + *(prand++) * lmin_sigma;
            while (lmax <= 0) lmax = lmax_center + *(prand++) * lmax_sigma;
        }
        while (w < 0 ) w = w_center + *(prand++) * w_sigma;
        wdir = fmod( wdir_center + *(prand++) * wdir_sigma, M_PI );
        while (z0 <= 0) z0 = z0_center + *(prand++) * z0_sigma;
        while (T0 <= 0) T0 = T0_center + *(prand++) * T0_sigma;
    }

    int ierr;

    ierr = MPI_Bcast( &lmin, 1, MPI_DOUBLE, 0, comm );
    if ( ierr != MPI_SUCCESS ) throw std::runtime_error( "Failed to bcast lmin" );

    ierr = MPI_Bcast( &lmax, 1, MPI_DOUBLE, 0, comm );
    if ( ierr != MPI_SUCCESS ) throw std::runtime_error( "Failed to bcast lmax" );

    ierr =MPI_Bcast( &w, 1, MPI_DOUBLE, 0, comm );
    if ( ierr != MPI_SUCCESS ) throw std::runtime_error( "Failed to bcast w" );

    ierr = MPI_Bcast( &wdir, 1, MPI_DOUBLE, 0, comm );
    if ( ierr != MPI_SUCCESS ) throw std::runtime_error( "Failed to bcast wdir" );

    ierr = MPI_Bcast( &z0, 1, MPI_DOUBLE, 0, comm );
    if ( ierr != MPI_SUCCESS ) throw std::runtime_error( "Failed to bcast z0" );

    ierr= MPI_Bcast( &T0, 1, MPI_DOUBLE, 0, comm );
    if ( ierr != MPI_SUCCESS ) throw std::runtime_error( "Failed to bcast T0" );

    wx = w * sin( wdir );
    wy = w * cos( wdir );

    // Use the absolute values of the wind components to simplify
    // translating the slab

    wx = fabs( wx );
    wy = fabs( wy );

    if ( rank == 0 && verbosity > 0 ) {
        std::cerr << std::endl;
        std::cerr << "Atmospheric realization parameters:" << std::endl;
        std::cerr << " lmin = " << lmin << " m" << std::endl;
        std::cerr << " lmax = " << lmax << " m" << std::endl;
        std::cerr << "    w = " << w << " m/s" << std::endl;
        std::cerr << "   wx = " << wx << " m/s" << std::endl;
        std::cerr << "   wy = " << wy << " m/s" << std::endl;
        std::cerr << " wdir = " << wdir << " radians" << std::endl;
        std::cerr << "   z0 = " << z0 << " m" << std::endl;
        std::cerr << "   T0 = " << T0 << " K" << std::endl;
    }

    return;
}


void toast::atm::sim::get_volume() {

    // Stationary volume

    delta_z = zmax;
    // Maximum distance observed through the simulated volume
    double maxdist = delta_z / sin(elmin);
    // Maximum distance to employ the near field simulation
    rnear = delta_z / sin(elmax) * fnear;
    // Maximum distance to employ the very near field simulation
    rverynear = rnear * fnear;
    // Volume length
    delta_x = maxdist * cos(elmin);
    // Volume width
    delta_y = delta_x * tan(delta_az / 2) * 2;

    // Telescope position wrt the full volume

    xtel = 0;
    ytel = delta_y / 2;
    ztel = 0;

    // Wind effect

    double wdx = wx * delta_t;
    double wdy = wy * delta_t;
    delta_x += wdx;
    delta_y += wdy;
    xtel += wdx;
    ytel += wdy;

    // Margin for interpolation

    delta_x += 5 * xstep;
    delta_y += ystep;
    delta_z += zstep;

    // Grid points

    nx = delta_x/xstep + 1;
    ny = delta_y/ystep + 1;
    nz = delta_z/zstep + 1;
    nn = nx * ny * nz;

    // 1D storage of the 3D volume elements

    zstride = 1;
    ystride = zstride * nz;
    xstride = ystride * ny;

    if ( rank == 0 && verbosity > 0 ) {
        std::cerr << std::endl;
        std::cerr << "Simulation volume:" << std::endl;
        std::cerr << "    height = " << delta_z << " m" << std::endl;
        std::cerr << "    length = " << delta_x << " m" << std::endl;
        std::cerr << "     width = " << delta_y << " m " << std::endl;
        std::cerr << "   maxdist = " << maxdist << " m" << std::endl;
        std::cerr << "     rnear = " << rnear << " m" << std::endl;
        std::cerr << " rverynear = " << rverynear << " m" << std::endl;
        std::cerr << "        nx = " << nx << std::endl;
        std::cerr << "        ny = " << ny << std::endl;
        std::cerr << "        nz = " << nz << std::endl;
        std::cerr << "        nn = " << nn << std::endl;
        std::cerr << "      xtel = " << xtel << " m" << std::endl;
        std::cerr << "      ytel = " << ytel << " m" << std::endl;
        std::cerr << "      ztel = " << ztel << " m" << std::endl;
    }

    initialize_kolmogorov();

}


void toast::atm::sim::initialize_kolmogorov() {

    double t1 = MPI_Wtime();

    // Numerically integrate the modified Kolmogorov spectrum for the
    // correlation function at grid points. We integrate down from
    // 10*kappamax to 0 for numerical precision

    rmin = 0;
    double diag = sqrt( delta_x*delta_x + delta_y*delta_y);
    rmax = sqrt( diag*diag + delta_z*delta_z );
    nr = 10000; // Size of the interpolation grid

#ifdef DEBUG
    nr /= 10;
#endif
    
    rstep = (rmax - rmin) / (nr-1);
    rstep_inv = 1. / rstep;

    kolmo_x.clear();
    kolmo_x.resize( nr, 0 );
    kolmo_y.clear();
    kolmo_y.resize( nr, 0 );

    double kappamin = 1. / lmax;
    double kappamax = 1. / lmin;
    double kappal = 0.9 * kappamax;
    double invkappal = 1 / kappal; // Optimize
    double kappa0 = 0.75 * kappamin;
    double kappa0sq = kappa0 * kappa0; // Optimize
    long nkappa = 1000000; // Number of integration steps needs to be large
    
#ifdef DEBUG
    std::cerr << "DEBUG = True: reducing kappa grid." << std::endl;
    nkappa /= 100;
#endif
    
    double upper_limit = 10*kappamax;
    double kappastep = upper_limit / (nkappa - 1);
    double slope1 = 7. / 6.;
    double slope2 = -11. / 6.;

    if ( rank == 0 && verbosity > 0 ) {
        std::cerr << std::endl;
        std::cerr << "Evaluating Kolmogorov correlation at " << nr
                  << " different separations in range " << rmin
                  << " - " << rmax << " m" << std::endl;
        std::cerr << "kappamin = " << kappamin
                  << " 1/m, kappamax =  " << kappamax
                  << " 1/m. nkappa = " << nkappa << std::endl;
    }
    
    // Use Newton's method to integrate the correlation function

    long nr_task = nr / ntask + 1;
    long first_r = nr_task * rank;
    long last_r = first_r + nr_task;
    if (last_r > nr) last_r = nr;

    // Precalculate the power spectrum function

    std::vector<double> phi(nkappa);
#pragma omp parallel for schedule(static, 10)
    for ( long ikappa=0; ikappa<nkappa; ++ikappa ) {
        double kappa = ikappa*kappastep;
        double kkl = kappa * invkappal;
        phi[ikappa] = ( 1. + 1.802 * kkl - 0.254 * pow( kkl, slope1 ) )
            * exp( -kkl*kkl ) * pow( kappa*kappa + kappa0sq, slope2 );
    }

    // Newton's method factors, not part of the power spectrum

    phi[0] /= 2;
    phi[nkappa-1] /= 2;

    // Integrate the power spectrum for a spherically symmetric
    // correlation function

#pragma omp parallel for schedule(static, 10)
    for ( long ir=first_r; ir<last_r; ++ir ) {
        double r = rmin + ir*rstep;
        double val = 0;
        if (r == 0) {
            // special limit r -> 0, sin(kappa.r)/r -> kappa
            for ( long ikappa=nkappa-1; ikappa>=0; --ikappa ) {
                double kappa = ikappa*kappastep;
                double kappasq = kappa * kappa;
                val += phi[ikappa] * kappasq;
            }
        } else {
            for ( long ikappa=nkappa-1; ikappa>=0; --ikappa ) {
                double kappa = ikappa*kappastep;
                val += phi[ikappa] * sin( kappa * r ) * kappa;
            }
            val /= r;
        }
        val *= kappastep;
        kolmo_x[ ir ] = r;
        kolmo_y[ ir ] = val;
    }

    int ierr;
    ierr = MPI_Allreduce( MPI_IN_PLACE, kolmo_x.data(), (int)nr,
                          MPI_DOUBLE, MPI_SUM, comm );
    if ( ierr != MPI_SUCCESS )
        throw std::runtime_error( "Failed to allreduce kolmo_x." );
    ierr = MPI_Allreduce( MPI_IN_PLACE, kolmo_y.data(), (int)nr,
                          MPI_DOUBLE, MPI_SUM, comm );
    if ( ierr != MPI_SUCCESS )
        throw std::runtime_error( "Failed to allreduce kolmo_y." );

    if ( rank == 0 && verbosity > 10) {
        std::ofstream f;
        std::ostringstream fname;
        fname << "kolmogorov.txt";
        f.open( fname.str(), std::ios::out );
        for ( int ir=0; ir<nr; ir++ )
            f << kolmo_x[ir] << " " << kolmo_y[ir] << std::endl;
        f.close();
    }

    double t2 = MPI_Wtime();

    if ( rank == 0 && verbosity > 0 )
        std::cerr << "Kolmogorov initialized in " << t2-t1 << " s." << std::endl;

    return;
}


double toast::atm::sim::kolmogorov( double r ) {

    // Return autocovariance of a Kolmogorov process at separation r

    if ( r == 0 ) return kolmo_y[0];
    if ( r == rmax ) return kolmo_y[nr-1];

    // Simple linear interpolation for now. Assume the r-grid is regular so
    // we don't need to search for the interpolation points.

    long ir = (r - rmin) * rstep_inv;

    if ( ir < 0 || ir > nr-2 ) {
        std::ostringstream o;
        o.precision( 16 );
        o << "Kolmogorov value requested at " << r
          << ", outside gridded range [" << rmin << ", " << rmax << "].";
        throw std::runtime_error( o.str().c_str() );
    }

    double rlow = kolmo_x[ir];
    double rhigh = kolmo_x[ir+1];
    double rdist = (r - rlow) / (rhigh - rlow);
    double vlow = kolmo_y[ir];
    double vhigh = kolmo_y[ir+1];

    double val = (1-rdist) * vlow + rdist * vhigh;

    return val;
}

  
void toast::atm::sim::compress_volume() {

    // Establish a mapping between full volume indices and observed volume indices

    double t1 = MPI_Wtime();

    compressed_index.resize( nn, -1 );
    full_index.resize( nn, -1 );

    std::vector<unsigned char> hit( nn, false );

    // Start by flagging all elements that are hit

    for (long ix=0; ix<nx-1; ++ix) {
        if ( ix % ntask != rank ) continue;
        double x = ix * xstep;

#pragma omp parallel for schedule(static, 10)
        for (long iy=0; iy<ny-1; ++iy) {
            double y = iy * ystep;

            for (long iz=0; iz<nz-1; ++iz) {
                double z = iz * zstep;
                if ( in_cone( x, y, z ) ) {
#ifdef DEBUG
                    hit.at( ix * xstride + iy * ystride + iz * zstride ) = true;
#else
                    hit[ ix * xstride + iy * ystride + iz * zstride ] = true;
#endif
                }
            }
        }
    }

    // For extra margin, flag all the neighbors of the hit elements

    std::vector<unsigned char> hit2 = hit;
#pragma omp parallel for schedule(static, 10)
    for (long ix=1; ix<nx-1; ++ix) {

        for (long iy=1; iy<ny-1; ++iy) {

            for (long iz=1; iz<nz-1; ++iz) {

                long offset = ix * xstride + iy * ystride + iz * zstride;

                if ( hit2[offset] ) {
                    // Flag this element but also its neighbours to facilitate
                    // interpolation

                    for ( double xmul=-1; xmul < 3; ++xmul ) {
                        if ( ix + xmul < 0 || ix + xmul > nx-1 ) continue;

                        for ( double ymul=-1; ymul < 3; ++ymul ) {
                            if ( iy + ymul < 0 || iy + ymul > ny-1 ) continue;

                            for ( double zmul=-1; zmul < 3; ++zmul ) {
                                if ( iz + zmul < 0 || iz + zmul > nz-1 ) continue;

#ifdef DEBUG
                                hit.at( offset + xmul*xstride
                                        + ymul*ystride + zmul*zstride ) = true;
#else
                                hit[ offset + xmul*xstride
                                     + ymul*ystride + zmul*zstride ] = true;
#endif
                            }
                        }
                    }
                }
            }
        }
    }

    hit2.resize(0);

    int ierr;
    ierr = MPI_Allreduce( MPI_IN_PLACE, hit.data(), (int)nn,
                          MPI_UNSIGNED_CHAR, MPI_LOR, comm );
    if ( ierr != MPI_SUCCESS )
        throw std::runtime_error( "Failed to gather hits" );

    // Then create the mappings between the compressed and full indices

    long i=0;
    for (long ifull=0; ifull<nn; ++ifull) {
#ifdef DEBUG
        if ( hit.at(ifull) ) {
            full_index.at(i) = ifull;
            compressed_index.at(ifull) = i;
            ++i;
        }
#else
        if ( hit[ifull] ) {
            full_index[i] = ifull;
            compressed_index[ifull] = i;
            ++i;
        }
#endif
    }

    nelem = i;

    full_index.resize( nelem );

    double t2 = MPI_Wtime();

    if ( rank == 0 and verbosity > 0 ) {
        std::cerr << "Volume compressed in " << t2-t1 << " s." << std::endl;
        std::cerr << i << " / " << nn
                  << " volume elements are needed for the simulation" << std::endl;
    }

    if ( nelem == 0 )
        throw std::runtime_error( "No elements in the observation cone." );

}


bool toast::atm::sim::in_cone( double x, double y, double z ) {

    if ( z >= zmax ) return false;

    // The wind makes the calculation rather involved. For now, we simply
    // perform a stationary in_cone check at a number of time points

    double tstep = 1;
    if ( wx != 0 ) tstep = xstep / wx;
    if ( wy != 0 )
        if ( tstep > ystep / wy) tstep = ystep / wy;
    tstep /= 10;
    long nt = delta_t / tstep + 1;
    if (nt < 1) nt = 1;

    for ( long it=0; it<nt; ++it ) {
        double t = it*tstep;
        double xtel_now = xtel - wx*t;
        double ytel_now = ytel - wy*t;

        double dxmin = x - xtel_now;
        double dxmax = dxmin + xstep;
        // Is the point is behind the telescope at this time?
        if ( dxmin < 0 && dxmax < 0 ) continue;

        double dymin = y - ytel_now;
        double dymax = dymin + ystep;

        double rmin = sqrt(dxmin*dxmin + dymin*dymin);
        double rmax = sqrt(dxmax*dxmax + dymax*dymax);
        double rmininv = 1 / rmin;
        double rmaxinv = 1 / rmax;

        // Is it at the observed elevation?

        double eltanmin = (z - zstep) * rmaxinv; // tangent of the xyz elevation
        if ( eltanmin > tanmax ) continue;

        double eltanmax = (z + zstep) * rmininv; // tangent of the xyz elevation
        if ( eltanmax < tanmin ) continue;

        // Is the data point in sector?

        double sinazmin = (dymin - ystep) * rmaxinv;
        if ( sinazmin > sinmax ) continue;

        double sinazmax = (dymax + ystep) * rmaxinv;
        if ( sinazmax < sinmin ) continue;

        return true;
    }

    return false;
}


void toast::atm::sim::ind2coord( long i, double *coord ) {

    // Translate a compressed index into xyz-coordinates

    long ifull = full_index[i];

    long ix = ifull / xstride;
    long iy = (ifull - ix*xstride) / ystride;
    long iz = ifull - ix*xstride - iy*ystride;

    coord[0] = ix * xstep;
    coord[1] = iy * ystep;
    coord[2] = iz * zstep;
}


long toast::atm::sim::coord2ind( double x, double y, double z ) {

    // Translate xyz-coordinates into a compressed index

    long ix = x / xstep;
    long iy = y / ystep;
    long iz = z / zstep;

    if ( ix < 0 || ix > nx-1 || iy < 0 || iy > ny-1 || iz < 0 || iz > nz-1 ) {
        std::ostringstream o;
        o.precision( 16 );
        o << "atmsim::coord2ind : full index out of bounds at ("
          << x << ", " << y << ", "<< z << ") = ("
          << ix << " /  " << nx << ", " << iy << " / " << ny << ", "
          << iz << ", " << nz << ")";
        throw std::runtime_error( o.str().c_str() );
    }

    size_t ifull = ix * xstride + iy * ystride + iz * zstride;

    return compressed_index[ifull];
}


double toast::atm::sim::interp( std::vector<double> &realization, double x, 
				double y, double z, std::vector<long> &last_ind,
				std::vector<double> &last_nodes ) {

    // Trilinear interpolation

    long ix = x * xstepinv;
    long iy = y * ystepinv;
    long iz = z * zstepinv;

    double dx = (x - (double)ix * xstep) * xstepinv;
    double dy = (y - (double)iy * ystep) * ystepinv;
    double dz = (z - (double)iz * zstep) * zstepinv;

    double c000, c001, c010, c011, c100, c101, c110, c111;

    if ( ix != last_ind[0] || iy != last_ind[1] || iz != last_ind[2] ) {

#ifdef DEBUG
        if ( ix < 0 || ix > nx-2 || iy < 0 || iy > ny-2 || iz < 0 || iz > nz-2 ) {
            std::ostringstream o;
            o.precision( 16 );
            o << "atmsim::interp : full index out of bounds at ("
              << x << ", " << y << ", "<< z << ") = ("
              << ix << "/" << nx << ", "
              << iy << "/" << ny << ", "
              << iz << "/" << nz << ")";
            throw std::runtime_error( o.str().c_str() );
        }
#endif

        size_t offset = ix * xstride + iy * ystride + iz * zstride;

        size_t ifull000 = offset;
        size_t ifull001 = offset + zstride;
        size_t ifull010 = offset + ystride;
        size_t ifull011 = ifull010 + zstride;
        size_t ifull100 = offset + xstride;
        size_t ifull101 = ifull100 + zstride;
        size_t ifull110 = ifull100 + ystride;
        size_t ifull111 = ifull110 + zstride;

#ifdef DEBUG
        long i000 = compressed_index.at(ifull000);
        long i001 = compressed_index.at(ifull001);
        long i010 = compressed_index.at(ifull010);
        long i011 = compressed_index.at(ifull011);
        long i100 = compressed_index.at(ifull100);
        long i101 = compressed_index.at(ifull101);
        long i110 = compressed_index.at(ifull110);
        long i111 = compressed_index.at(ifull111);
#else
        long i000 = compressed_index[ifull000];
        long i001 = compressed_index[ifull001];
        long i010 = compressed_index[ifull010];
        long i011 = compressed_index[ifull011];
        long i100 = compressed_index[ifull100];
        long i101 = compressed_index[ifull101];
        long i110 = compressed_index[ifull110];
        long i111 = compressed_index[ifull111];
#endif

        if (i001 < 0) i001 = i000;
        if (i011 < 0) i011 = i010;
        if (i101 < 0) i101 = i100;
        if (i111 < 0) i111 = i110;

#ifdef DEBUG
        c000 = realization.at(i000);
        c001 = realization.at(i001);
        c010 = realization.at(i010);
        c011 = realization.at(i011);
        c100 = realization.at(i100);
        c101 = realization.at(i101);
        c110 = realization.at(i110);
        c111 = realization.at(i111);
#else
        c000 = realization[i000];
        c001 = realization[i001];
        c010 = realization[i010];
        c011 = realization[i011];
        c100 = realization[i100];
        c101 = realization[i101];
        c110 = realization[i110];
        c111 = realization[i111];
#endif

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


El::DistMatrix<double> *toast::atm::sim::build_covariance( long ind_start,
							   long ind_stop,
							   bool save_covmat,
							   double scale ) {

    double t1 = MPI_Wtime();

    // Allocate the distributed matrix

    long nelem_slice = ind_stop - ind_start;

    El::DistMatrix<double> *cov = NULL;

    try {
        // Distributed element-element covariance matrix
        cov = new El::DistMatrix<double>( nelem_slice, nelem_slice, *grid );
    } catch ( std::bad_alloc & e ) {
        std::cerr << rank << " : Out of memory allocating covariance." << std::endl;
        throw;
    }

    // Report memory usage

    double my_mem = cov->AllocatedMemory() * 2 * sizeof(double) / pow(2.0, 20);
    double tot_mem;
    int ierr = MPI_Allreduce( &my_mem, &tot_mem, 1, MPI_DOUBLE, MPI_SUM,
                              comm_gang );
    if ( ierr != MPI_SUCCESS )
        throw std::runtime_error( "Failed to allreduce covariance matrix size." );
    if ( rank_gang == 0 && verbosity > 0 ) {
        std::cerr << std::endl;
        std::cerr << "Gang # " << gang << " Allocated " << tot_mem
                  << " MB for the distributed covariance matrix." << std::endl;
    }

    // Fill the elements of the covariance matrix. Each task populates the matrix
    // elements that are already stored locally.

    int nrow = cov->LocalHeight();
    int ncol = cov->LocalWidth();

    for (int icol=0; icol<ncol; ++icol) {
        for (int irow=0; irow<nrow; ++irow) {

            // Translate local indices to global indices

            int globalcol = cov->GlobalCol( icol );
            int globalrow = cov->GlobalRow( irow );

            // Translate global indices into coordinates

            double colcoord[3], rowcoord[3];
            ind2coord( globalcol + ind_start, colcoord );
            ind2coord( globalrow + ind_start, rowcoord );

            // Evaluate the covariance between the two coordinates

            double val = cov_eval(colcoord, rowcoord, scale);
            cov->SetLocal( irow, icol, val );
        }
    }

    double t2 = MPI_Wtime();

    if ( rank == 0 && verbosity > 0 )
        std::cerr << "Gang # "<< gang << " Covariance constructed in "
                  << t2-t1 << " s." << std::endl;

    if ( save_covmat ) {
        std::ostringstream fname;
        fname << "covariance_" << ind_start << "_" << ind_stop;
        El::Write( *cov, fname.str(), El::BINARY_FLAT );
    }

    return cov;
}


double toast::atm::sim::cov_eval( double *coord1, double *coord2,
				  double scale ) {

    // Evaluate the atmospheric absorption covariance between two coordinates
    // Church (1995) Eq.(6) & (9)

    double x1=coord1[0], y1=coord1[1], z1=coord1[2];
    double x2=coord2[0], y2=coord2[1], z2=coord2[2];

    // Water vapor altitude factor

    double chi1 = exp( -(z1+z2) * scale / (2*z0) );

    // Kolmogorov factor

    double dx = x1-x2, dy=y1-y2, dz=z1-z2;
    double r = sqrt( dx*dx + dy*dy + dz*dz ) * scale;
    double chi2 = kolmogorov( r );

    return chi1 * chi2;
}


void toast::atm::sim::sqrt_covariance( El::DistMatrix<double> *cov, 
				       long ind_start, long ind_stop,
				       bool save_covmat, int near ) {

    // Cholesky decompose the covariance matrix.  If the matrix is singular,
    // regularize it by adding power to the diagonal.

    long nelem_slice = ind_stop - ind_start;

    for ( int attempt=0; attempt < 10; ++attempt ) {
        try {
            El::DistMatrix<double> cov_temp(*cov);

            MPI_Barrier( comm_gang );
            double t1 = MPI_Wtime();

            if ( rank_gang == 0 && verbosity > 0 ) {
                std::cerr << std::endl;
                std::cerr << "Gang # " << gang << " near = " << near
                          << " Cholesky decomposing covariance ... "
                          << std::endl;
            }

            El::Cholesky( El::UPPER, cov_temp );

            MPI_Barrier( comm_gang );
            double t2 = MPI_Wtime();

            if ( rank_gang == 0 && verbosity > 0 ) {
                std::cerr << std::endl;
                std::cerr << "Gang # " << gang << " near = " << near
                          << " Cholesky decomposition done in " << t2-t1 << " s. N = "
                          << nelem_slice << " ntask = " << ntask << " nthread = "
                          << nthread << std::endl;
            }

            *cov = cov_temp;

            break;

        } catch ( ... ) {

            if ( rank_gang == 0 && verbosity > 0 ) {
                std::cerr << std::endl;
                std::cerr << "Gang # " << gang << " near = " << near
                          << " Cholesky decomposition failed on attempt " << attempt
                          << ". Regularizing matrix. " << std::endl;
                if ( attempt == 9 ) {
                    //El::Write( *cov, "failed_covmat", El::BINARY_FLAT ); // DEBUG
                    throw std::runtime_error( "Failed to decompose covariance matrix." );
                }
            }

            int nrow = cov->LocalHeight();
            int ncol = cov->LocalWidth();

            for (int icol=0; icol<ncol; ++icol) {
                for (int irow=0; irow<nrow; ++irow) {

                    // Translate local indices to global indices

                    int globalcol = cov->GlobalCol( icol );
                    int globalrow = cov->GlobalRow( irow );

                    if ( globalcol == globalrow ) {
                        // Double the diagonal value
                        double val = cov->GetLocal( irow, icol );
                        cov->SetLocal( irow, icol, val * 2 );
                    }
                }
            }
        }
    }

    if ( save_covmat ) {
        std::ostringstream fname;
        fname << "sqrt_covariance_" << near << "_" << ind_start << "_" << ind_stop;
        El::Write( *cov, fname.str(), El::BINARY_FLAT );
    }

    return;
}


void toast::atm::sim::apply_covariance( El::DistMatrix<double> *cov,
					std::vector<double> &realization,
					long ind_start, long ind_stop,
					int near ) {

    double t1 = MPI_Wtime();

    long nelem_slice = ind_stop - ind_start;

    // Generate a realization of the atmosphere that matches the structure
    // of the element-element covariance matrix.  The covariance matrix in
    // "cov" is assumed to have been replaced with its square root.

    // Draw the Gaussian variates in a single call

    size_t nrand = nelem_slice;
    std::vector<double> randn(nrand);
    rng::dist_normal( nrand, key1, key2, counter1, counter2, randn.data() );
    counter2 += nrand;
    double *prand=randn.data();

    // Atmosphere realization

    El::DistMatrix<double,El::STAR,El::STAR> slice_realization( *grid );
    El::Zeros( slice_realization, 1, nelem_slice );

    // Instantiate a realization with gaussian random variables on root process

    if (rank_gang == 0) {
        slice_realization.Reserve( nelem_slice );
        for ( int col=0; col<nelem_slice; ++col ) {
            slice_realization.QueueUpdate( 0, col, *(prand++) );
        }
    } else {
        slice_realization.Reserve( 0 );
    }

    slice_realization.ProcessQueues();

    if ( rank_gang == 0 && verbosity > 10 ) {
        double *p = slice_realization.Buffer();

        std::ofstream f;
        std::ostringstream fname;
        fname << "raw_realization_" << near << "_"
              << ind_start << "_" << ind_stop << ".txt";
        f.open( fname.str(), std::ios::out );
        for ( long ielem=0; ielem<nelem_slice; ielem++ ) {
            double coord[3];
            ind2coord( ielem, coord );
            f << coord[0] << " " << coord[1] << " " << coord[2] << " "
              << p[ielem]  << std::endl;
        }
        f.close();
    }

    // Apply the sqrt covariance to impose correlations

    // Atmosphere realization

    // El::Trmv is not implemented yet in Elemental
    //El::Trmv( El::UPPER, El::NORMAL, El::NON_UNIT, *cov, slice_realization );
    El::Trmm( El::LEFT, El::UPPER, El::NORMAL, El::NON_UNIT,
              1.0, *cov, slice_realization );

    // Subtract the mean of the slice to reduce step between the slices

    double *p = slice_realization.Buffer();
    double mean = 0, var = 0;
    for ( long i=0; i<nelem_slice; ++i ) {
        mean += p[i];
        var += p[i] * p[i];
    }
    mean /= nelem_slice;
    var = var / nelem_slice - mean*mean;
    for ( long i=0; i<nelem_slice; ++i ) p[i] -= mean;

    double t2 = MPI_Wtime();

    if ( rank_gang == 0 && verbosity > 0 ) {
        std::cerr << std::endl;
        std::cerr << "Gang # " << gang << " near = " << near
                  << " Realization slice (" << ind_start << " -- " << ind_stop
                  << ") var = " << var << ", constructed in "
                  << t2-t1 << " s." << std::endl;
    }

    if ( rank_gang == 0 && verbosity > 10 ) {
        std::ofstream f;
        std::ostringstream fname;
        fname << "realization_" << near << "_"
              << ind_start << "_" << ind_stop << ".txt";
        f.open( fname.str(), std::ios::out );
        for ( long ielem=0; ielem<nelem_slice; ielem++ ) {
            double coord[3];
            ind2coord( ielem, coord );
            f << coord[0] << " " << coord[1] << " " << coord[2] << " "
              << p[ielem]  << std::endl;
        }
        f.close();
    }

    // Copy the slice realization over appropriate indices in the full realization
    // FIXME: This is where we would blend slices

    for ( long i=ind_start; i<ind_stop; ++i ) {
        realization[i] = p[i-ind_start];
    }

    return;
}
