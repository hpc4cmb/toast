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


namespace toast { namespace atm {

#ifdef HAVE_ELEMENTAL

    class atmsim {

        public :

            // The atmsim object is constructed for one CES (constant elevation scan)

            atmsim( double azmin, double azmax, double elmin, double elmax, double tmin, double tmax, // CES parameters
                double lmin_center=.01, double lmin_sigma=.001, // dissipation scale of the Kolmogorov turbulence
                double lmax_center=10, double lmax_sigma=10, // injection scale of the Kolmogorov turbulence
                double w_center=25, double w_sigma=10, // wind speed [m/s]
                double wdir_center=0, double wdir_sigma=100, // wind direction [radians]
                double z0_center=2000, double z0_sigma=0, // Water vapor distribution [m]
                double T0_center=280, double T0_sigma=10, // Ground temperature
                double zatm=40000, // Atmosphere extent for temperature profile
                double zmax=2000, // Water vapor extent for integration
                double xstep=100, double ystep=100, double zstep=100, // Size of the volume elements
                long nelem_sim_max=1000, // Controls the size of the simulation slices
                int verbosity=0, MPI_Comm comm=MPI_COMM_WORLD,
                int gangsize=-1, // Size of the gangs that create slices
                double fnear=0.1 ); // Multiplier for the near field simulation

            ~atmsim();

            // we can simulate a number of realizations for the same CES and distribution of parameters
            void simulate( bool save_covmat );

            // ::observe can only be called after ::simulate and only with compatible arguments.
            void observe( double *t, double *az, double *el, double *tod, long nsamp, double fixed_r=-1 );

        private :

            MPI_Comm comm, comm_gang;
            int rank, ntask, rank_gang, ntask_gang, nthread, gangsize, gang, ngang, verbosity;
            double azmin, azmax, elmin, elmax, tmin, tmax;
            double tanmin, tanmax, sinmin, sinmax; // In-cone calculation helpers
            double lmin_center, lmin_sigma, lmax_center, lmax_sigma,
            w_center, w_sigma, wdir_center, wdir_sigma,
            z0_center, z0_sigma, T0_center, T0_sigma;
            double az0, delta_az, delta_el, delta_t;
            double zatm, zmax;
            double xstep, ystep, zstep, delta_x, delta_y, delta_z, xstepinv, ystepinv, zstepinv;
            double fnear, fnearinv, rnear, rverynear;
            long nx, ny, nz, nn, nelem, xstride, ystride, zstride;
            double xtel, ytel, ztel; // Telescope position
            double lmin, lmax, w, wdir, z0, T0, wx, wy;
            long nr; // Number of steps in the Kolmogorov grid
            long nelem_sim_max; // Sets the size of the independent X-direction slices.
            double rmin, rmax, rstep, rstep_inv; // Kolmogorov correlation grid
            std::vector<long> compressed_index; // Mapping between full volume and observation cone
            std::vector<long> full_index; // Inverse mapping between full volume and observation cone
            std::vector<double> atmosphere; // The actual realization
            void draw(); // Draw values of lmin, lmax, w, wdir, T0 (and optionally z0)
            void get_volume(); // Determine the rectangular volume needed
            bool in_cone(double x, double y, double z); // determine of the given coordinates are within observed volume
            void compress_volume(); // Find the volume elements really needed
            El::Grid *grid=NULL;
            El::DistMatrix<double> *cov=NULL;
            std::vector<double> realization, realization_near, realization_verynear;
            void get_slice( long &ind_start, long &ind_stop ); // Find the next range of compressed indices to simulate
            void build_covariance( long ind_start, long ind_stop, bool save_covmat, double scale ); // Use the atmospheric parameters for volume element covariance
            void sqrt_covariance( El::DistMatrix<double> *cov, long ind_start, long ind_stop, bool save_covmat=false, int near=0 ); // Take the (regularized) square root of the covariance matrix
            void apply_covariance( El::DistMatrix<double> *cov, std::vector<double> &realization, long ind_start, long ind_stop, int near=0 ); // Create a realization out of the square root covariance matrix
            void ind2coord( long i, double *coord ); // Compressed index to xyz-coordinates
            long coord2ind( double x, double y, double z ); // xyz-coordinates to Compressed index
            double interp( std::vector<double> &realization, double x, double y, double z, std::vector<long> &last_ind, std::vector<double> &last_nodes ); // Interpolate the realization value to given coordinates
            double cov_eval( double *coord1, double *coord2, double scale=1 ); // Evaluate the covariance matrix
            void initialize_kolmogorov(); // Integrate the correlation from the Kolmorov spectrum
            double kolmogorov( double r ); // Interpolate the correlation from precomputed grid
            void smooth(); // Smooth the realization
            std::vector<double> kolmo_x;
            std::vector<double> kolmo_y;

    };

#endif

} }


#endif
