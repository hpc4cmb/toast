/*
  Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
  All rights reserved.  Use of this source code is governed by
  a BSD-style license that can be found in the LICENSE file.
*/

#if !defined(DEBUG)
#   define DEBUG
#endif

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
    if ( vec.size() == 0 ) return 0;

    sort( vec.begin(), vec.end());
    int half1 = (vec.size() - 1) * .5;
    int half2 = vec.size() * .5;

    return .5 *( vec[half1] + vec[half2] );
}


double mean( std::vector<double> vec ) {
    if ( vec.size() == 0 ) return 0;

    double sum = 0;
    for ( auto& val : vec ) sum += val;

    return sum / vec.size();
}


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
              int verbosity, MPI_Comm comm, int gangsize,
              uint64_t key1,uint64_t key2,
              uint64_t counter1, uint64_t counter2, char *cachedir)
: comm(comm), cachedir(cachedir),
  gangsize(gangsize), verbosity(verbosity),
  key1(key1), key2(key2), counter1start(counter1), counter2start(counter2),
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
  nelem_sim_max(nelem_sim_max)
{
    counter1 = counter1start;
    counter2 = counter2start;

    if ( MPI_Comm_size( comm, &ntask ) )
        throw std::runtime_error( "Failed to get size of MPI communicator." );
    if ( MPI_Comm_rank( comm, &rank ) )
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
        if ( MPI_Comm_split( comm, gang, rank, &comm_gang ) )
            throw std::runtime_error( "Failed to split MPI communicator." );
        if ( MPI_Comm_size( comm_gang, &ntask_gang ) )
            throw std::runtime_error( "Failed to get size of the split MPI "
                                      "communicator." );
        if ( MPI_Comm_rank( comm_gang, &rank_gang ) )
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

    delta_az = ( azmax - azmin );
    delta_el = ( elmax - elmin );
    delta_t = tmax - tmin;

    az0 = azmin + delta_az / 2;
    el0 = elmin + delta_el / 2;
    sinel0 = sin( el0 );
    cosel0 = cos( el0 );

    xxstep = xstep*cosel0 - zstep*sinel0;
    yystep = ystep;
    zzstep = xstep*sinel0 + zstep*cosel0;

    // speed up the in-cone calculation
    double tol = 0.1 * M_PI / 180; // 0.1 degree tolerance
    tanmin = tan( -0.5*delta_az - tol );
    tanmax = tan(  0.5*delta_az + tol );

    if ( rank == 0 && verbosity > 0 ) {
        std::cerr << std::endl;
        std::cerr << "Input parameters:" << std::endl;
        std::cerr << "             az = [" << azmin*180./M_PI << " - "
                  << azmax*180./M_PI << "] (" << delta_az*180./M_PI
                  << " degrees)" << std::endl;
        std::cerr << "             el = [" << elmin*180./M_PI << " - "
                  << elmax*180./M_PI << "] (" << delta_el*180/M_PI
                  << " degrees)" << std::endl;
        std::cerr << "              t = [" << tmin << " - " << tmax
                  << "] (" << delta_t << " s)" << std::endl;
        std::cerr << "           lmin = " << lmin_center << " +- " << lmin_sigma
                  << " m" << std::endl;
        std::cerr << "           lmax = " << lmax_center << " +- " << lmax_sigma
                  << " m" << std::endl;
        std::cerr << "              w = " << w_center << " +- " << w_sigma
                  << " m" << std::endl;
        std::cerr << "           wdir = " << wdir_center*180./M_PI << " +- "
                  << wdir_sigma*180./M_PI << " degrees " << std::endl;
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
    if ( compressed_index ) delete compressed_index;
    if ( full_index ) delete full_index;
    if ( realization ) delete realization;
    if ( comm_gang != MPI_COMM_NULL && comm_gang != MPI_COMM_SELF
         && comm_gang != comm ) {
        if ( MPI_Comm_free( &comm_gang ) )
            throw std::runtime_error( "Failed to free MPI communicator." );
    }
}

void toast::atm::sim::load_realization() {

    cached = false;

    std::ostringstream name;
    name << key1 << "_" << key2 << "_"
         << counter1start << "_" << counter2start;

    char success;

    if ( rank == 0 ) {

        // Load metadata

        success = 1;

        std::ostringstream fname;
        fname << cachedir << "/" << name.str() << "_metadata.txt";

        std::ifstream f(fname.str());
        if ( f.good() ) {
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

            if ( rank == 0 and verbosity > 0 ) {
                std::cerr << "Loaded metada from "
                          << fname.str() << std::endl;
            }
        } else
            success = 0;

        if ( verbosity > 0 && success ) {
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
            std::cerr << " wdir = " << wdir*180./M_PI << " degrees" << std::endl;
            std::cerr << "   z0 = " << z0 << " m" << std::endl;
            std::cerr << "   T0 = " << T0 << " K" << std::endl;
        }
    }

    if ( MPI_Bcast( &success, 1, MPI_CHAR, 0, comm ) )
        throw std::runtime_error( "Failed to bcast success" );

    if ( !success ) return;

    if ( MPI_Bcast( &nn, 1, MPI_LONG, 0, comm ) )
        throw std::runtime_error( "Failed to bcast nn" );
    if ( MPI_Bcast( &nelem, 1, MPI_LONG, 0, comm ) )
        throw std::runtime_error( "Failed to bcast nn" );
    if ( MPI_Bcast( &nx, 1, MPI_LONG, 0, comm ) )
        throw std::runtime_error( "Failed to bcast nx" );
    if ( MPI_Bcast( &ny, 1, MPI_LONG, 0, comm ) )
        throw std::runtime_error( "Failed to bcast ny" );
    if ( MPI_Bcast( &nz, 1, MPI_LONG, 0, comm ) )
        throw std::runtime_error( "Failed to bcast nz" );
    if ( MPI_Bcast( &delta_x, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast delta_x" );
    if ( MPI_Bcast( &delta_y, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast delta_y" );
    if ( MPI_Bcast( &delta_z, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast delta_z" );
    if ( MPI_Bcast( &xstart, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast xstart" );
    if ( MPI_Bcast( &ystart, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast ystart" );
    if ( MPI_Bcast( &zstart, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast zstart" );
    if ( MPI_Bcast( &maxdist, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast maxdist" );
    if ( MPI_Bcast( &wx, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast wx" );
    if ( MPI_Bcast( &wy, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast wy" );
    if ( MPI_Bcast( &wz, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast wz" );
    if ( MPI_Bcast( &lmin, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast lmin" );
    if ( MPI_Bcast( &lmax, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast lmax" );
    if ( MPI_Bcast( &w, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast w" );
    if ( MPI_Bcast( &wdir, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast wdir" );
    if ( MPI_Bcast( &z0, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast z0" );
    if ( MPI_Bcast( &T0, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast T0" );

    zstride = 1;
    ystride = zstride * nz;
    xstride = ystride * ny;

    xstrideinv = 1. / xstride;
    ystrideinv = 1. / ystride;
    zstrideinv = 1. / zstride;

    // Load realization

    try {
        compressed_index = new mpi_shmem_long( nn, comm );
        compressed_index->set( -1 );

        full_index = new mpi_shmem_long( nelem, comm );
        full_index->set( -1 );
    } catch ( std::bad_alloc & e ) {
        std::cerr << rank
                  << " : Out of memory allocating element indices. nn = "
                  << nn << std::endl;
        throw;
    }

    try {
        realization = new mpi_shmem_double( nelem, comm );
        realization->set( 0 );
    } catch ( std::bad_alloc & e ) {
        std::cerr << rank
                  << " : Out of memory allocating realization. nelem = "
                  << nelem << std::endl;
        throw;
    }

    if ( full_index->rank() == 0 ) {
        std::ostringstream fname_real;
        fname_real << cachedir << "/" << name.str() << "_realization.dat";
        std::ifstream freal( fname_real.str(),
                             std::ios::in | std::ios::binary);

        freal.read( (char*) &(*full_index)[0],
                    full_index->size()*sizeof(long) );
        for ( int i=0; i<nelem; ++i ) {
            long ifull = (*full_index)[i];
            (*compressed_index)[ifull] = i;
        }

        freal.read( (char*) &(*realization)[0],
                    realization->size()*sizeof(double) );

        freal.close();

        if ( verbosity > 0 )
            std::cerr << "Loaded realization from "
                      << fname_real.str() << std::endl;
    }

    cached = true;

    return;
}

void toast::atm::sim::save_realization() {

    if ( rank == 0 ) {

        std::ostringstream name;
        name << key1 << "_" << key2 << "_"
             << counter1start << "_" << counter2start;

        // Save metadata

        std::ostringstream fname;
        fname << cachedir << "/" << name.str() << "_metadata.txt";

        std::ofstream f;
        f.precision( 16 );
        f.open( fname.str() );
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

        // Save realization

        std::ostringstream fname_real;
        fname_real << cachedir << "/" << name.str() << "_realization.dat";
        std::ofstream freal( fname_real.str(),
                             std::ios::out | std::ios::binary);

        freal.write( (char*) &(*full_index)[0],
                      full_index->size()*sizeof(long) );

        freal.write( (char*) &(*realization)[0],
                     realization->size()*sizeof(double) );

        freal.close();

        if ( verbosity > 0 )
            std::cerr << "Saved realization to "
                      << fname_real.str() << std::endl;
    }

    return;
}

void toast::atm::sim::simulate( bool use_cache ) {

    if ( use_cache ) load_realization();

    if ( cached ) return;

    try {

        draw();

        get_volume();

        compress_volume();

        if ( rank == 0 and verbosity > 0 ) {
            std::cerr << "Resizing realization to " << nelem << std::endl;
        }

        try {
            realization = new mpi_shmem_double( nelem, comm );
            realization->set( 0 );
        } catch ( std::bad_alloc & e ) {
            std::cerr << rank
                      << " : Out of memory allocating realization. nelem = "
                      << nelem << std::endl;
            throw;
        }

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

                El::DistMatrix<double> *cov = build_covariance(
                    ind_start, ind_stop );
                sqrt_covariance( cov, ind_start, ind_stop );
                apply_covariance( cov, ind_start, ind_stop );

                delete cov;
            }

            if ( ind_stop == nelem ) break;

            ++slice;
        }

        // Gather the slices from the gangs

        for ( size_t slice=0; slice < slice_starts.size(); ++slice ) {
            ind_start = slice_starts[slice];
            ind_stop = slice_stops[slice];
            int nind = ind_stop - ind_start;
            int root_gang = slice % ngang;
            int root = root_gang * gangsize;
            std::vector<double> tempvec(nind);
            if ( rank == root ) {
                std::memcpy( tempvec.data(), realization->data()+ind_start,
                             sizeof(double) * nind );
            }
            if ( MPI_Bcast( tempvec.data(), nind, MPI_DOUBLE, root, comm ) ) {
                throw std::runtime_error("Failed to broadcast the realization");
            }
            if ( realization->rank() == 0 ) {
                std::memcpy( realization->data()+ind_start, tempvec.data(),
                             sizeof(double) * nind );
            }
        }

        //smooth();

        MPI_Barrier( comm );
        double t2 = MPI_Wtime();

        if ( rank == 0 && verbosity > 0 ) {
            std::cerr << std::endl;
            std::cerr << "Realization constructed in " << t2-t1 << " s."
                      << std::endl;
        }

    } catch ( const std::exception& e ) {
        std::cerr << "WARNING: atm::simulate failed with: " << e.what()
                  << std::endl;
    }

    cached = true;

    if ( use_cache ) save_realization();

    return;
}


void toast::atm::sim::get_slice( long &ind_start, long &ind_stop ) {

    // Identify a manageable slice of compressed indices to simulate next

    ind_start = ind_stop;

    long ix_start = (*full_index)[ind_start] * xstrideinv;
    long ix1 = ix_start;
    long ix2;

    while ( true ) {
        ix2 = ix1;
        while ( ix1 == ix2 ) {
            ++ind_stop;
            if ( ind_stop == nelem ) break;
            ix2 = (*full_index)[ind_stop] * xstrideinv;
        }
        if ( ind_stop == nelem ) break;
        if ( ind_stop - ind_start > nelem_sim_max ) break;
        ix1 = ix2;
    }

    if ( rank == 0 && verbosity > 0 ) {
        std::cerr << "X-slice: " << ix_start*xstep << " -- " << ix2*xstep
                  << " m out of  " << nx*xstep << " m"
                  << " indices " << ind_start << " -- "<< ind_stop
                  << " ( " << ind_stop - ind_start << " )"
                  << " out of " << nelem << std::endl;
    }

    return;
}


void toast::atm::sim::smooth() {

    // Replace each vertex with a mean of its immediate vicinity

    double t1 = MPI_Wtime();

    double coord[3];

    std::vector<double> smoothed_realization(realization->size());

    for ( size_t i=0; i < full_index->size(); ++i ) {
        ind2coord( i, coord );
        long ix = coord[0] * xstepinv;
        long iy = coord[1] * ystepinv;
        long iz = coord[2] * zstepinv;

        long offset = ix * xstride + iy * ystride + iz * zstride;

        long w = 3; // width of the smoothing kernel
        long ifullmax = compressed_index->size();

        std::vector<double> vals;

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

                    long ifull = offset + xoff*xstride + yoff*ystride
                        + zoff*zstride;

                    if ( ifull < 0 || ifull >= ifullmax )
                        throw std::runtime_error(
                            "Index out of range in smoothing." );

                    long ii = (*compressed_index)[ifull];

                    if (ii >= 0) {
                        vals.push_back( (*realization)[ii] );
                    }
                }
            }
        }

        // Get the smoothed value

        smoothed_realization[i] = mean( vals );
    }

    if ( realization->rank() == 0 ) {
        for ( int i=0; i<realization->size(); ++i ) {
            (*realization)[i] = smoothed_realization[i];
        }
    }

    double t2 = MPI_Wtime();

    if ( rank == 0 && verbosity > 0 )
        std::cerr << "Realization smoothed in " << t2-t1 << " s." << std::endl;

    return;
}


void toast::atm::sim::observe( double *t, double *az, double *el, double *tod,
			       long nsamp, double fixed_r ) {

    if ( !cached ) {
        throw std::runtime_error( "There is no cached observation to observe" );
    }

    try {

        double t1 = MPI_Wtime();

        // For each sample, integrate along the line of sight by summing
        // the atmosphere values. See Church (1995) Section 2.2, first equation.
        // We omit the optical depth factor which is close to unity.

        double zatm_inv = 1. / zatm;

#pragma omp parallel for schedule(static, 100)
        for ( long i=0; i<nsamp; ++i ) {

            if ( az[i] < azmin || az[i] > azmax
                 || el[i] < elmin || el[i] > elmax ) {
                std::ostringstream o;
                o.precision( 16 );
                o << "atmsim::observe : observation out of bounds (az, el, t)"
                  << " = (" << az[i] << ",  " << el[i] << ", " << t[i]
                  << ") allowed: (" << azmin << " - "<< azmax << ", "
                  << elmin << " - "<< elmax << ", "
                  << tmin << " - "<< tmax << ")"
                  << std::endl;
                throw std::runtime_error( o.str().c_str() );
            }

            double t_now = t[i] - tmin;
            double az_now = az[i] - az0; // Relative to center of field
            double el_now = el[i];

            double xtel_now = wx*t_now;
            double ytel_now = wy*t_now;
            double ztel_now = wz*t_now;

            double sin_el = sin( el_now );
            double sin_el_max = sin( elmax );
            double cos_el = cos( el_now );
            double sin_az = sin( az_now );
            double cos_az = cos( az_now );

            // We want to choose rstart and rstep so that we exactly
            // sample the volume at the center of the volume elements in
            // the X (in scan) direction.

            /*
              double rstart = 10;
              double dr = 1;
              double dz = dr * sin_el;
              double drproj = dr * cos_el;
              double dx = drproj * cos_az;
              double dxx = dx*cosel0 + dz*sinel0;
              double rstep = xstep / dxx;

              double r = rstart; // Start integration at a reasonable distance

              double z = r * sin_el;
              double rproj = r * cos_el;
              double x = xtel_now + rproj*cos_az;
              double xx = x*cosel0 + z*sinel0;
              long ix = (xx-xstart) * xstepinv;
              double frac = (xx - (xstart + (double)ix*xstep)) * xstepinv;
              frac += .5;
              r += (1-frac) * rstep;
            */

            double r = 1.5 * xstep;
            double rstep = xstep;

            std::vector<long> last_ind(3);
            std::vector<double> last_nodes(8);

            double val = 0;
            if ( fixed_r > 0 ) r = fixed_r;

            while ( true ) {

                // Coordinates at distance r. The scan is centered on the X-axis

                // Check if the top of the focal plane hits zmax at
                // this distance.  This way all lines-of-sight get
                // integrated to the same distance
                double zz = r * sin_el_max;
                if ( zz >= zmax ) break;

                // Horizontal coordinates

                zz = r * sin_el;
                double rproj = r * cos_el;
                double xx = rproj * cos_az;
                double yy = -rproj * sin_az;

                // Rotate to scan frame

                double x = xx*cosel0 + zz*sinel0;
                double y = yy;
                double z = -xx*sinel0 + zz*cosel0;

                // Translate by the wind

                x += xtel_now;
                y += ytel_now;
                z += ztel_now;

#ifdef DEBUG
                if ( x < xstart || x > xstart+delta_x ||
                     y < ystart || y > ystart+delta_y ||
                     z < zstart || z > zstart+delta_z ) {
                    //std::ostringstream o;
                    //o.precision( 16 );
                    std::cerr << "atmsim::observe : (x,y,z) out of bounds: "
                              << std::endl
                              << "x = " << x << std::endl
                              << "y = " << y << std::endl
                              << "z = " << z << std::endl;
                    //throw std::runtime_error( o.str().c_str() );
                    if ( x < 0 ) x = xstart;
                    if ( x > xstart+delta_x ) x = xstart+delta_x;
                    if ( y < 0 ) y = ystart;
                    if ( y > ystart+delta_y ) y = ystart+delta_y;
                    if ( z < 0 ) z = xstart;
                    if ( z > zstart+delta_z ) z = zstart+delta_z;
                }
#endif
                // Combine atmospheric emission (via interpolation) with the
                // ambient temperature

                double step_val;
                try {
                    step_val = interp( x, y, z, last_ind, last_nodes )
                        * (1. - z * zatm_inv);
                } catch ( const std::runtime_error& e ) {
                    std::ostringstream o;
                    o.precision( 16 );
                    o << "atmsim::observe : interp failed at " << std::endl
                      << "xxyyzz = (" << xx << ", " << yy << ", " << zz << ")"
                      << std::endl
                      << "xyz = (" << x << ", " << y << ", " << z << ")"
                      << std::endl
                      << "r = " << r << std::endl
                      << "tele at (" << xtel_now << ", " << ytel_now << ", "
                      << ztel_now << ")" << std::endl
                      << "( t, az, el ) = " << "( " << t[i]-tmin << ", "
                      << az_now*180/M_PI
                      << " deg , " << el_now*180/M_PI << " deg) "
                      << " in_cone(t) = " << in_cone( x, y, z, t_now )
                      << " with "
                      << std::endl << e.what() << std::endl;
                    throw std::runtime_error( o.str().c_str() );
                }

                // Add dissipation due to distance

                step_val /= r*r;

                val += step_val;

                // Prepare for the next step

                r += rstep;

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
        std::cerr << "WARNING: atm::observe failed with: " << e.what()
                  << std::endl;
    } catch ( ... ) {
        std::cerr << "WARNING: atm::observe failed with an unknown exception."
                  << std::endl;
    }

    return;
}


void toast::atm::sim::draw() {

    // Draw 100 gaussian variates to use in drawing the simulation parameters

    const size_t nrand = 10000;
    double randn[nrand];
    rng::dist_normal( nrand, key1, key2, counter1, counter2, randn );
    counter2 += nrand;
    double *prand=randn;
    long irand=0;

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
            while (lmin <= 0 && irand < nrand-1)
                lmin = lmin_center + randn[irand++] * lmin_sigma;
            while (lmax <= 0 && irand < nrand-1)
                lmax = lmax_center + randn[irand++] * lmax_sigma;
        }
        while (w < 0 && irand < nrand-1)
            w = w_center + randn[irand++] * w_sigma;
        wdir = fmod( wdir_center + randn[irand++] * wdir_sigma, M_PI );
        while (z0 <= 0 && irand < nrand)
            z0 = z0_center + randn[irand++] * z0_sigma;
        while (T0 <= 0 && irand < nrand)
            T0 = T0_center + randn[irand++] * T0_sigma;

        if (irand == nrand)
            throw std::runtime_error(
                "Failed to draw parameters so satisfy boundary conditions" );
    }

    if ( MPI_Bcast( &lmin, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast lmin" );

    if ( MPI_Bcast( &lmax, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast lmax" );

    if ( MPI_Bcast( &w, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast w" );

    if ( MPI_Bcast( &wdir, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast wdir" );

    if ( MPI_Bcast( &z0, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast z0" );

    if ( MPI_Bcast( &T0, 1, MPI_DOUBLE, 0, comm ) )
        throw std::runtime_error( "Failed to bcast T0" );

    // Precalculate the ratio for covariance

    z0inv = 1. / (2. * z0);

    // Wind parallel to surface

    double wx_h = w * sin( wdir );
    wy = w * cos( wdir );

    // Rotate to a frame where scan is along X axis

    wx = wx_h * cosel0;
    wz = -wx_h * sinel0;

    if ( rank == 0 && verbosity > 0 ) {
        std::cerr << std::endl;
        std::cerr << "Atmospheric realization parameters:" << std::endl;
        std::cerr << " lmin = " << lmin << " m" << std::endl;
        std::cerr << " lmax = " << lmax << " m" << std::endl;
        std::cerr << "    w = " << w << " m/s" << std::endl;
        std::cerr << "   wx = " << wx << " m/s" << std::endl;
        std::cerr << "   wy = " << wy << " m/s" << std::endl;
        std::cerr << "   wz = " << wz << " m/s" << std::endl;
        std::cerr << " wdir = " << wdir*180./M_PI << " degrees" << std::endl;
        std::cerr << "   z0 = " << z0 << " m" << std::endl;
        std::cerr << "   T0 = " << T0 << " K" << std::endl;
    }

    return;
}


void toast::atm::sim::get_volume() {

    // Horizontal volume

    double delta_z_h = zmax;
    //std::cerr << "delta_z_h = " << delta_z_h << std::endl;

    // Maximum distance observed through the simulated volume
    maxdist = delta_z_h / sinel0;
    //std::cerr << "maxdist = " << maxdist << std::endl;

    // Volume length
    double delta_x_h = maxdist * cos(elmin);
    //std::cerr << "delta_x_h = " << delta_x_h << std::endl;

    double x, y, z, xx, zz, r, rproj, z_min, z_max;
    r = maxdist;

    z = r * sin(elmin);
    rproj = r * cos(elmin);
    x = rproj * cos(0);
    z_min = -x*sinel0 + z*cosel0;

    z = r * sin(elmax);
    rproj = r * cos(elmax);
    x = rproj * cos(delta_az/2);
    z_max = -x*sinel0 + z*cosel0;

    // Cone width
    delta_y_cone = maxdist * tan(delta_az / 2.) * 2.;
    //std::cerr << "delta_y_cone = " << delta_y_cone << std::endl;

    // Cone height
    delta_z_cone = z_max - z_min;
    //std::cerr << "delta_z_cone = " << delta_z_cone << std::endl;

    // Rotate to observation plane

    delta_x = maxdist;
    //std::cerr << "delta_x = " << delta_x << std::endl;
    delta_z = delta_z_cone;
    //std::cerr << "delta_z = " << delta_z << std::endl;

    // Wind effect

    double wdx = std::abs(wx) * delta_t;
    double wdy = std::abs(wy) * delta_t;
    double wdz = std::abs(wz) * delta_t;
    delta_x += wdx;
    delta_y = delta_y_cone + wdy;
    delta_z += wdz;

    // Margin for interpolation

    delta_x += xstep;
    delta_y += 2*ystep;
    delta_z += 2*zstep;

    // Translate the volume to allow for wind.  Telescope sits
    // at (0, 0, 0) at t=0

    if ( wx < 0 )
        xstart = -wdx;
    else
        xstart = 0;

    if ( wy < 0 )
        ystart = -0.5*delta_y_cone - wdy - ystep;
    else
        ystart = -0.5*delta_y_cone - ystep;

    if ( wz < 0 )
        zstart = z_min - wdz - zstep;
    else
        zstart = z_min - zstep;

    // Grid points

    nx = delta_x/xstep + 1;
    ny = delta_y/ystep + 1;
    nz = delta_z/zstep + 1;
    nn = nx * ny * nz;

    // 1D storage of the 3D volume elements

    zstride = 1;
    ystride = zstride * nz;
    xstride = ystride * ny;

    xstrideinv = 1. / xstride;
    ystrideinv = 1. / ystride;
    zstrideinv = 1. / zstride;

    if ( rank == 0 && verbosity > 0 ) {
        std::cerr << std::endl;
        std::cerr << "Simulation volume:" << std::endl;
        std::cerr << "   delta_x = " << delta_x << " m" << std::endl;
        std::cerr << "   delta_y = " << delta_y << " m" << std::endl;
        std::cerr << "   delta_z = " << delta_z << " m" << std::endl;
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


void toast::atm::sim::initialize_kolmogorov() {

    MPI_Barrier( comm );
    double t1 = MPI_Wtime();

    // Numerically integrate the modified Kolmogorov spectrum for the
    // correlation function at grid points. We integrate down from
    // 10*kappamax to 0 for numerical precision

    rmin = 0;
    double diag = sqrt( delta_x*delta_x + delta_y*delta_y);
    rmax = sqrt( diag*diag + delta_z*delta_z ) * 1.01;
    nr = 1000; // Size of the interpolation grid

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

    long nkappa_task = nkappa / ntask + 1;
    long first_kappa = nkappa_task * rank;
    long last_kappa = first_kappa + nkappa_task;
    if (last_kappa > nkappa) last_kappa = nkappa;

    // Precalculate the power spectrum function

    std::vector<double> phi(last_kappa - first_kappa);
#pragma omp parallel for schedule(static, 10)
    for ( long ikappa=first_kappa; ikappa<last_kappa; ++ikappa ) {
        double kappa = ikappa*kappastep;
        double kkl = kappa * invkappal;
        phi[ikappa-first_kappa] =
            ( 1. + 1.802 * kkl - 0.254 * pow( kkl, slope1 ) )
            * exp( -kkl*kkl ) * pow( kappa*kappa + kappa0sq, slope2 );
    }

    /*
      if ( rank == 0 && verbosity > 0) {
      std::ofstream f;
      std::ostringstream fname;
      fname << "kolmogorov_f.txt";
      f.open( fname.str(), std::ios::out );
      for ( int ikappa=0; ikappa<nkappa; ++ikappa )
      f << ikappa*kappastep << " " << phi[ikappa] << std::endl;
      f.close();
      }
    */

    // Newton's method factors, not part of the power spectrum

    if ( first_kappa == 0 ) phi[0] /= 2;
    if ( last_kappa == nkappa ) phi[last_kappa-first_kappa-1] /= 2;

    // Integrate the power spectrum for a spherically symmetric
    // correlation function

    double nri = 1. / (nr-1);
    double tau = 10.;
    double enorm = 1. / (exp(tau) - 1.);
    double ifac3 = 1. / (2.*3.);

#pragma omp parallel for schedule(static, 10)
    for ( long ir=0; ir<nr; ++ir ) {
        double r = rmin + (exp(ir*nri*tau)-1)*enorm*(rmax-rmin);
        double val = 0;
        if ( r * kappamax < 1e-2 ) {
            // special limit r -> 0,
            // sin(kappa.r)/r -> kappa - kappa^3*r^2/3!
            for ( long ikappa=first_kappa; ikappa<last_kappa; ++ikappa ) {
                double kappa = ikappa*kappastep;
                double kappa2 = kappa * kappa;
                double kappa4 = kappa2 * kappa2;
                double r2 = r * r;
                val += phi[ikappa - first_kappa] * (kappa2 - r2*kappa4*ifac3);
            }
        } else {
            for ( long ikappa=first_kappa; ikappa<last_kappa; ++ikappa ) {
                double kappa = ikappa*kappastep;
                val += phi[ikappa - first_kappa] * sin( kappa * r ) * kappa;
            }
            val /= r;
        }
        val *= kappastep;
        kolmo_x[ ir ] = r;
        kolmo_y[ ir ] = val;
    }

    if ( MPI_Allreduce( MPI_IN_PLACE, kolmo_y.data(), (int)nr,
                        MPI_DOUBLE, MPI_SUM, comm ) )
        throw std::runtime_error( "Failed to allreduce kolmo_y." );

    // Normalize

    double norm = 1. / kolmo_y[0];
    for ( int i=0; i<nr; ++i ) kolmo_y[i] *= norm;

    if ( rank == 0 && verbosity > 0) {
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

    if ( r < rmin || r > rmax ) {
        std::ostringstream o;
        o.precision( 16 );
        o << "Kolmogorov value requested at " << r
          << ", outside gridded range [" << rmin << ", " << rmax << "].";
        throw std::runtime_error( o.str().c_str() );
    }

    // Simple linear interpolation for now.  Use a bisection method
    // to find the rigth elements.

    long low = 0, high = nr-1;
    long ir;

    while ( true ) {
        ir = low + 0.5*(high-low);
        if (kolmo_x[ir] <= r and r <= kolmo_x[ir+1]) break;
        if (r < kolmo_x[ir])
            high = ir;
        else
            low = ir;
    }

    double rlow = kolmo_x[ir];
    double rhigh = kolmo_x[ir+1];
    double rdist = (r - rlow) / (rhigh - rlow);
    double vlow = kolmo_y[ir];
    double vhigh = kolmo_y[ir+1];

    double val = (1-rdist)*vlow + rdist*vhigh;

    return val;
}


void toast::atm::sim::compress_volume() {

    // Establish a mapping between full volume indices and observed
    // volume indices

    double t1 = MPI_Wtime();

    std::vector<unsigned char> hit;
    try {
        compressed_index = new mpi_shmem_long( nn, comm );
        compressed_index->set( -1 );

        full_index = new mpi_shmem_long( nn, comm );
        full_index->set( -1 );

        hit.resize( nn, false );
    } catch ( std::bad_alloc & e ) {
        std::cerr << rank
                  << " : Out of memory allocating element indices. nn = "
                  << nn << std::endl;
        throw;
    }

    // Start by flagging all elements that are hit

    for ( long ix=0; ix<nx-1; ++ix ) {
        if ( ix % ntask != rank ) continue;
        double x = xstart + ix*xstep;

#pragma omp parallel for schedule(static, 10)
        for ( long iy=0; iy<ny-1; ++iy ) {
            double y = ystart + iy*ystep;

            for ( long iz=0; iz<nz-1; ++iz ) {
                double z = zstart + iz*zstep;
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

    if ( MPI_Allreduce( MPI_IN_PLACE, hit.data(), (int)nn,
                        MPI_UNSIGNED_CHAR, MPI_LOR, comm ) )
        throw std::runtime_error( "Failed to gather hits" );

    std::vector<unsigned char> hit2 = hit;

    for (long ix=1; ix<nx-1; ++ix) {
        if ( ix % ntask != rank ) continue;

#pragma omp parallel for schedule(static, 10)
        for (long iy=1; iy<ny-1; ++iy) {

            for (long iz=1; iz<nz-1; ++iz) {

                long offset = ix * xstride + iy * ystride + iz * zstride;

                if ( hit2[offset] ) {
                    // Flag this element but also its neighbours to facilitate
                    // interpolation

                    for ( double xmul=-2; xmul < 4; ++xmul ) {
                        if ( ix + xmul < 0 || ix + xmul > nx-1 ) continue;

                        for ( double ymul=-2; ymul < 4; ++ymul ) {
                            if ( iy + ymul < 0 || iy + ymul > ny-1 ) continue;

                            for ( double zmul=-2; zmul < 4; ++zmul ) {
                                if ( iz + zmul < 0 || iz + zmul > nz-1 )
                                    continue;

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

    if ( MPI_Allreduce( MPI_IN_PLACE, hit.data(), (int)nn,
                        MPI_UNSIGNED_CHAR, MPI_LOR, comm ) )
        throw std::runtime_error( "Failed to gather hits" );

    // Then create the mappings between the compressed and full indices

    long i=0;
    for (long ifull=0; ifull<nn; ++ifull) {
        if ( hit[ifull] ) {
            (*full_index)[i] = ifull;
            (*compressed_index)[ifull] = i;
            ++i;
        }
    }

    hit.resize(0);
    nelem = i;

    full_index->resize( nelem );

    double t2 = MPI_Wtime();

    if ( rank == 0 ) {
        //if ( verbosity > 0 ) {
        std::cout << "Volume compressed in " << t2-t1 << " s." << std::endl
                  << i << " / " << nn << "(" << i * 100. / nn << " %)"
                  << " volume elements are needed for the simulation"
                  << std::endl
                  << "nx = " << nx << " ny = " << ny << " nz = " << nz
                  << std::endl
                  << "wx = " << wx << " wy = " << wy << " wz = " << wz
                  << std::endl;
        //}
    }

    if ( nelem == 0 )
        throw std::runtime_error( "No elements in the observation cone." );

}


bool toast::atm::sim::in_cone( double x, double y, double z, double t_in ) {

    // Input coordinates are in the scan frame, rotate to horizontal frame

    double tstep = 1;

    for ( double t=0; t<delta_t; t+=tstep ) {
        if ( t_in >= 0 ) {
            if ( t != 0 ) break;
            t = t_in;
        }

        if ( t_in < 0 && delta_t - t < tstep ) t = delta_t;

        double xtel_now = wx * t;
        double dx = x - xtel_now;

        // Is the point behind the telescope at this time?

        if ( dx + xstep < 0 ) {
            if ( t_in >= 0 ) std::cerr << "dx + xstep < 0: " << dx << std::endl;
            continue;
        }

        // Check the rest of the spherical coordinates

        double ytel_now = wy * t;
        double dy = y - ytel_now;

        double ztel_now = wz * t;
        double dz = z - ztel_now;

        double r = std::sqrt( dx*dx + dy*dy + dz*dz );
        if ( r > maxdist*1.01 ) {
            if ( t_in >= 0 ) std::cerr << "r > maxdist " << r << std::endl;
            continue;
        }

        if (dz > 0)
            dz -= zstep;
        else
            dz += zstep;

        if ( std::abs(dy) < 2*ystep && std::abs(dz) < 2*zstep )
            return true;

        double dxx = dx*cosel0 - dz*sinel0;
        double dyy = dy;
        double dzz = dx*sinel0 + dz*cosel0;

        double el = std::asin( dzz / r );
        if ( el < elmin || el > elmax ) {
            if ( t_in >= 0 )
                std::cerr << "el outside cone: "
                          << el*180/M_PI << " not in "
                          << elmin*180/M_PI << " - "
                          << elmax*180/M_PI << std::endl;
            continue;
        }

        dxx = (dx+xstep)*cosel0 - dz*sinel0;
        double az = std::atan2( dyy, dxx );
        if ( std::abs(az) > 0.5*delta_az ) {
            if ( t_in >= 0 )
                std::cerr << "abs(az) > delta_az/2 "
                          << az*180/M_PI << " > "
                          << 0.5*delta_az*180/M_PI << std::endl;
            continue;
        }

        // Passed all the checks

        return true;
    }

    return false;
}


void toast::atm::sim::ind2coord( long i, double *coord ) {

    // Translate a compressed index into xyz-coordinates
    // in the horizontal frame

    long ifull = (*full_index)[i];

    long ix = ifull * xstrideinv;
    long iy = (ifull - ix*xstride) * ystrideinv;
    long iz = ifull - ix*xstride - iy*ystride;

    // coordinates in the scan frame

    double x = xstart + ix*xstep;
    double y = ystart + iy*ystep;
    double z = zstart + iz*zstep;

    // Into the horizontal frame

    coord[0] = x*cosel0 - z*sinel0;
    coord[1] = y;
    coord[2] = x*sinel0 + z*cosel0;

}


long toast::atm::sim::coord2ind( double x, double y, double z ) {

    // Translate scan frame xyz-coordinates into a compressed index

    long ix = (x-xstart) * xstepinv;
    long iy = (y-ystart) * ystepinv;
    long iz = (z-zstart) * zstepinv;

#ifdef DEBUG
    if ( ix < 0 || ix > nx-1 || iy < 0 || iy > ny-1 || iz < 0 || iz > nz-1 ) {
        std::ostringstream o;
        o.precision( 16 );
        o << "atmsim::coord2ind : full index out of bounds at ("
          << x << ", " << y << ", "<< z << ") = ("
          << ix << " /  " << nx << ", " << iy << " / " << ny << ", "
          << iz << ", " << nz << ")";
        throw std::runtime_error( o.str().c_str() );
    }
#endif

    size_t ifull = ix * xstride + iy * ystride + iz * zstride;

    return (*compressed_index)[ifull];
}


double toast::atm::sim::interp( double x, double y, double z,
                                std::vector<long> &last_ind,
				std::vector<double> &last_nodes ) {

    // Trilinear interpolation

    long ix = (x-xstart) * xstepinv;
    long iy = (y-ystart) * ystepinv;
    long iz = (z-zstart) * zstepinv;

    double dx = (x - (xstart + (double)ix*xstep)) * xstepinv;
    double dy = (y - (ystart + (double)iy*ystep)) * ystepinv;
    double dz = (z - (zstart + (double)iz*zstep)) * zstepinv;

#ifdef DEBUG
    if ( dx < 0 || dx > 1 || dy < 0 || dy > 1 || dz < 0 || dz > 1 ) {
        //std::ostringstream o;
        //o.precision( 16 );
        std::cerr << "atmsim::interp : bad fractional step: " << std::endl
                 << "x = " << x << std::endl
                 << "y = " << y << std::endl
                 << "z = " << z << std::endl
                 << "dx = " << dx << std::endl
                 << "dy = " << dy << std::endl
                 << "dz = " << dz << std::endl;
        //throw std::runtime_error( o.str().c_str() );
        if ( dx < 0 ) dx = 0;
        if ( dx > 1 ) dx = 1;
        if ( dy < 0 ) dy = 0;
        if ( dy > 1 ) dy = 1;
        if ( dz < 0 ) dz = 0;
        if ( dz > 1 ) dz = 1;
    }
#endif

    double c000, c001, c010, c011, c100, c101, c110, c111;

    if ( ix != last_ind[0] || iy != last_ind[1] || iz != last_ind[2] ) {

#ifdef DEBUG
        if ( ix < 0 || ix > nx-2 || iy < 0 || iy > ny-2
             || iz < 0 || iz > nz-2 ) {
            //std::ostringstream o;
            //o.precision( 16 );
            std::cerr << "atmsim::interp : full index out of bounds at"
                      << std::endl << "("
                      << x << ", " << y << ", "<< z << ") = ("
                      << ix << "/" << nx << ", "
                      << iy << "/" << ny << ", "
                      << iz << "/" << nz << ")";
            //throw std::runtime_error( o.str().c_str() );
            if ( ix < 0 ) ix = 0;
            if ( ix > nx-2 ) ix = nx-2;
            if ( iy < 0 ) iy = 0;
            if ( iy > ny-2 ) iy = ny-2;
            if ( iz < 0 ) iz = 0;
            if ( iz > nz-2 ) iz = nz-2;
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
        long ifullmax = compressed_index->size()-1;
        if (
            ifull000 < 0 || ifull000 > ifullmax ||
            ifull001 < 0 || ifull001 > ifullmax ||
            ifull010 < 0 || ifull010 > ifullmax ||
            ifull011 < 0 || ifull011 > ifullmax ||
            ifull100 < 0 || ifull100 > ifullmax ||
            ifull101 < 0 || ifull101 > ifullmax ||
            ifull110 < 0 || ifull110 > ifullmax ||
            ifull111 < 0 || ifull111 > ifullmax ) {
            std::ostringstream o;
            o.precision( 16 );
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
            throw std::runtime_error( o.str().c_str() );
        }
#endif

        long i000 = (*compressed_index)[ifull000];
        long i001 = (*compressed_index)[ifull001];
        long i010 = (*compressed_index)[ifull010];
        long i011 = (*compressed_index)[ifull011];
        long i100 = (*compressed_index)[ifull100];
        long i101 = (*compressed_index)[ifull101];
        long i110 = (*compressed_index)[ifull110];
        long i111 = (*compressed_index)[ifull111];

#ifdef DEBUG
        long imax = realization->size()-1;
        if (
            i000 < 0 || i000 > imax ||
            i001 < 0 || i001 > imax ||
            i010 < 0 || i010 > imax ||
            i011 < 0 || i011 > imax ||
            i100 < 0 || i100 > imax ||
            i101 < 0 || i101 > imax ||
            i110 < 0 || i110 > imax ||
            i111 < 0 || i111 > imax ) {
            //std::ostringstream o;
            //o.precision( 16 );
            //o << "atmsim::observe : bad compressed index. "
            std::cerr << "atmsim::observe : bad compressed index. "
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
                      << "in_cone(x, y, z) = " << in_cone( x, y, z )
                      << std::endl;
            //throw std::runtime_error( o.str().c_str() );
            int good=0;
            if ( i000 >= 0 && i000 < imax ) {
                good = i000;
            } else if ( i001 >= 0 && i001 < imax ) {
                good = i001;
            } else if ( i010 >= 0 && i010 < imax ) {
                good = i010;
            } else if ( i011 >= 0 && i011 < imax ) {
                good = i011;
            } else if ( i100 >= 0 && i100 < imax ) {
                good = i100;
            } else if ( i101 >= 0 && i101 < imax ) {
                good = i101;
            } else if ( i110 >= 0 && i110 < imax ) {
                good = i110;
            } else if ( i111 >= 0 && i111 < imax ) {
                good = i111;
            }
            if ( i000 < 0 || i000 > imax ) i000 = good;
            if ( i001 < 0 || i001 > imax ) i001 = good;
            if ( i010 < 0 || i010 > imax ) i010 = good;
            if ( i011 < 0 || i011 > imax ) i011 = good;
            if ( i100 < 0 || i100 > imax ) i100 = good;
            if ( i101 < 0 || i101 > imax ) i101 = good;
            if ( i110 < 0 || i110 > imax ) i110 = good;
            if ( i111 < 0 || i111 > imax ) i111 = good;
        }
#endif

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


El::DistMatrix<double> *toast::atm::sim::build_covariance(
    long ind_start, long ind_stop ) {

    double t1 = MPI_Wtime();

    // Allocate the distributed matrix

    long nelem_slice = ind_stop - ind_start;

    El::DistMatrix<double> *cov = NULL;

    try {
        // Distributed element-element covariance matrix
        cov = new El::DistMatrix<double>( nelem_slice, nelem_slice, *grid );
    } catch ( std::bad_alloc & e ) {
        std::cerr << rank << " : Out of memory allocating covariance."
                  << std::endl;
        throw;
    }

    // Report memory usage

    double my_mem = cov->AllocatedMemory() * 2 * sizeof(double) / pow(2.0, 20.0);
    double tot_mem;
    if ( MPI_Allreduce( &my_mem, &tot_mem, 1, MPI_DOUBLE, MPI_SUM, comm_gang ) )
        throw std::runtime_error(
            "Failed to allreduce covariance matrix size." );
    if ( rank_gang == 0 && verbosity > 0 ) {
        std::cerr << std::endl;
        std::cerr << "Gang # " << gang << " Allocated " << tot_mem
                  << " MB for the distributed covariance matrix." << std::endl;
    }

    // Fill the elements of the covariance matrix. Each task populates the matrix
    // elements that are already stored locally.

    int nrow = cov->LocalHeight();
    int ncol = cov->LocalWidth();

#pragma omp parallel for schedule(static, 10)
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

            double val = cov_eval( colcoord, rowcoord );
            cov->SetLocal( irow, icol, val );
        }
    }

    double t2 = MPI_Wtime();

    if ( rank == 0 && verbosity > 0 )
        std::cerr << "Gang # "<< gang << " Covariance constructed in "
                  << t2-t1 << " s." << std::endl;

    if ( false ) {
        std::ostringstream fname;
        fname << "covariance_" << ind_start << "_" << ind_stop;
        El::Write( *cov, fname.str(), El::BINARY_FLAT );
    }

    return cov;
}


double toast::atm::sim::cov_eval( double *coord1, double *coord2 ) {

    // Evaluate the atmospheric absorption covariance between two coordinates
    // Church (1995) Eq.(6) & (9)
    // Coordinates are in the horizontal frame

    const long nn = 1;
    const double ndxinv = xxstep / (nn-1);
    const double ndzinv = zzstep / (nn-1);
    const double ninv = 1. / ( nn * nn );

    double val = 0;

    for ( int ii1=0; ii1<nn; ++ii1 ) {
        double xx1 = coord1[0];
        double yy1 = coord1[1];
        double zz1 = coord1[2];

        if ( ii1 ) {
            xx1 += ii1 * ndxinv;
            zz1 += ii1 * ndzinv;
        }

        for ( int ii2=0; ii2<nn; ++ii2 ) {
            double xx2 = coord2[0];
            double yy2 = coord2[1];
            double zz2 = coord2[2];

            if ( ii2 ) {
                xx2 += ii2 * ndxinv;
                zz2 += ii2 * ndzinv;
            }

            // Water vapor altitude factor

            double chi1 = std::exp( -(zz1+zz2) * z0inv );

            // Kolmogorov factor

            double dx = xx1 - xx2;
            double dy = yy1 - yy2;
            double dz = zz1 - zz2;


            double r = sqrt( dx*dx + dy*dy + dz*dz );
            double chi2 = kolmogorov( r );

            val += chi1 * chi2;
        }
    }

    return val * ninv;
}


void toast::atm::sim::sqrt_covariance( El::DistMatrix<double> *cov,
				       long ind_start, long ind_stop ) {

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
                std::cerr << "Gang # " << gang
                          << " Cholesky decomposing covariance ... "
                          << std::endl;
            }

            El::Cholesky( El::LOWER, cov_temp );

            MPI_Barrier( comm_gang );
            double t2 = MPI_Wtime();

            if ( rank_gang == 0 && verbosity > 0 ) {
                std::cerr << std::endl;
                std::cerr << "Gang # " << gang
                          << " Cholesky decomposition done in " << t2-t1
                          << " s. N = " << nelem_slice
                          << " ntask_gang = " << ntask_gang
                          << " nthread = " << nthread << std::endl;
            }

            *cov = cov_temp;

            break;

        } catch ( ... ) {

            if ( rank_gang == 0 && verbosity > 0 ) {
                std::cerr << std::endl;
                std::cerr << "Gang # " << gang
                          << " Cholesky decomposition failed on attempt "
                          << attempt
                          << ". Regularizing matrix. " << std::endl;
                if ( attempt == 9 ) {
                    throw std::runtime_error(
                        "Failed to decompose covariance matrix." );
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

    if ( false ) {
        std::ostringstream fname;
        fname << "sqrt_covariance_" << ind_start << "_" << ind_stop;
        El::Write( *cov, fname.str(), El::BINARY_FLAT );
    }

    return;
}


void toast::atm::sim::apply_covariance( El::DistMatrix<double> *cov,
					long ind_start, long ind_stop ) {

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
        fname << "raw_realization_"
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
    //El::Trmm( El::LEFT, El::LOWER, El::NORMAL, El::NON_UNIT,
    //          1.0, *cov, slice_realization );
    // For some reason, multiplying from the left only used the
    // diagonal elements.
    El::Trmm( El::RIGHT, El::LOWER, El::TRANSPOSE, El::NON_UNIT,
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
        std::cerr << "Gang # " << gang
                  << " Realization slice (" << ind_start << " -- " << ind_stop
                  << ") var = " << var << ", constructed in "
                  << t2-t1 << " s." << std::endl;
    }

    if ( rank_gang == 0 && verbosity > 10 ) {
        std::ofstream f;
        std::ostringstream fname;
        fname << "realization_"
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

    // Copy the slice realization over appropriate indices in
    // the full realization
    // FIXME: This is where we would blend slices

    for ( long i=ind_start; i<ind_stop; ++i ) {
        (*realization)[i] = p[i-ind_start];
    }

    return;
}
