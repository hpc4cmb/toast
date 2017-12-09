/*
  Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
  All rights reserved.  Use of this source code is governed by
  a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_ATM_INTERNAL_HPP
#define TOAST_ATM_INTERNAL_HPP

#include <config.h>
#include <toast/atm.hpp>

#include <mpi.h>
#include <vector>

#ifdef HAVE_ELEMENTAL
#include <El.hpp>
#endif


namespace toast { namespace tatm {

typedef mpi_shmem::mpi_shmem<double> mpi_shmem_double;
typedef mpi_shmem::mpi_shmem<long> mpi_shmem_long;

#ifdef HAVE_AATM
double get_absorption_coefficient(double altitude, double pwv, double freq);
double get_atmospheric_loading(double altitude, double pwv, double freq);
#endif

#ifdef HAVE_ELEMENTAL

class sim {

public :

    // The sim object is constructed for one CES (constant elevation scan)

    sim( double azmin, double azmax, // CES azimuth range
         double elmin, double elmax, // CES elevation range
         double tmin, double tmax, // CES time range
         // dissipation scale of the Kolmogorov turbulence
         double lmin_center=.01, double lmin_sigma=.001,
         // injection scale of the Kolmogorov turbulence
         double lmax_center=10, double lmax_sigma=10,
         double w_center=25, double w_sigma=10, // wind speed [m/s]
         // wind direction [radians]
         double wdir_center=0, double wdir_sigma=100,
         // Water vapor distribution [m]
         double z0_center=2000, double z0_sigma=0,
         double T0_center=280, double T0_sigma=10, // Ground temperature
         double zatm=40000, // Atmosphere extent for temperature profile
         double zmax=2000, // Water vapor extent for integration
         // Size of the volume elements
         double xstep=100, double ystep=100, double zstep=100,
         long nelem_sim_max=1000, // Size of the simulation slices
         int verbosity=0, MPI_Comm comm=MPI_COMM_WORLD,
         int gangsize=-1, // Size of the gangs that create slices
         uint64_t key1=0, uint64_t key2=0, // RNG keys
         uint64_t counterval1=0, uint64_t counterval2=0, // RNG counters
         char *cachedir=NULL );

    ~sim();

    // we can simulate a number of realizations for the same CES
    // and distribution of parameters
    void simulate( bool use_cache );

    // ::observe can only be called after ::simulate and only with
    // compatible arguments.
    void observe( double *t, double *az, double *el, double *tod,
                  long nsamp, double fixed_r=-1 );

    void print();

private :

    MPI_Comm comm=MPI_COMM_NULL, comm_gang=MPI_COMM_NULL;
    std::string cachedir;
    int rank, ntask, rank_gang, ntask_gang, nthread, gangsize, gang, ngang;
    int verbosity;
    uint64_t key1, key2, counter1, counter2, counter1start, counter2start;
    double azmin, azmax, elmin, elmax, tmin, tmax, sinel0, cosel0;
    double tanmin, tanmax; // In-cone calculation helpers
    double lmin_center, lmin_sigma, lmax_center, lmax_sigma,
        w_center, w_sigma, wdir_center, wdir_sigma,
        z0_center, z0_sigma, T0_center, T0_sigma, z0inv;
    double az0, el0, delta_az, delta_el, delta_t;
    double zatm, zmax;
    double xstep, ystep, zstep, delta_x, delta_y, delta_z;
    double xstart, ystart, zstart, xxstep, yystep, zzstep;
    double delta_y_cone, delta_z_cone, maxdist;
    double xstepinv, ystepinv, zstepinv;
    long nx, ny, nz, nn, xstride, ystride, zstride;
    double xstrideinv, ystrideinv, zstrideinv;
    size_t nelem;
    bool cached=false;
    double lmin, lmax, w, wdir, z0, T0, wx, wy, wz;
    long nr; // Number of steps in the Kolmogorov grid
    long nelem_sim_max; // Size of the independent X-direction slices.
    double rmin, rmax, rstep, rstep_inv; // Kolmogorov correlation
                                         // grid
    // Mapping between full volume and observation cone
    mpi_shmem_long *compressed_index=NULL;
    // Inverse mapping between full volume and observation cone
    mpi_shmem_long *full_index=NULL;
    void draw(); // Draw values of lmin, lmax, w, wdir, T0 (and optionally z0)
    void get_volume(); // Determine the rectangular volume needed
    // determine of the given coordinates are within observed volume
    bool in_cone( double x, double y, double z, double t_in=-1 );
    void compress_volume(); // Find the volume elements really needed
    El::Grid *grid=NULL;
    mpi_shmem_double *realization=NULL;
    // Find the next range of compressed indices to simulate
    void get_slice( long &ind_start, long &ind_stop );
    // Use the atmospheric parameters for volume element covariance
    El::DistMatrix<double> * build_covariance( long ind_start, long ind_stop );
    // Cholesky decompose (square root) the covariance matrix
    void sqrt_covariance( El::DistMatrix<double> *cov, long ind_start,
                          long ind_stop );
    // Create a realization out of the square root covariance matrix
    void apply_covariance( El::DistMatrix<double> *cov,
                           long ind_start, long ind_stop );
    // Compressed index to xyz-coordinates
    void ind2coord( long i, double *coord );
    // xyz-coordinates to Compressed index
    long coord2ind( double x, double y, double z );
    // Interpolate the realization value to given coordinates
    double interp( double x, double y, double z, std::vector<long> &last_ind,
                   std::vector<double> &last_nodes );
    // Evaluate the covariance matrix
    double cov_eval( double *coord1, double *coord2 );
    // Integrate the correlation from the Kolmorov spectrum
    void initialize_kolmogorov();
    // Interpolate the correlation from precomputed grid
    double kolmogorov( double r );
    void smooth(); // Smooth the realization
    std::vector<double> kolmo_x;
    std::vector<double> kolmo_y;
    void load_realization();
    void save_realization();
};

#endif

} }


#endif
