
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_ATM_HPP
#define TOAST_ATM_HPP

#ifdef HAVE_CHOLMOD

extern "C" {
# include <cholmod.h>
}

namespace toast {
// This small singleton class is used to initialize and finalize the cholmod package.

class CholmodCommon {
    public:

        // Singleton access
        static CholmodCommon & get();
        ~CholmodCommon();

        cholmod_common cholcommon;
        cholmod_common * chcommon;

    private:

        // This class is a singleton- constructor is private.
        CholmodCommon();
};

bool atm_verbose();

bool atm_sim_in_cone(
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
    );

void atm_sim_compress_flag_hits_rank(
    long nn,
    uint8_t * hit,
    int ntask,
    int rank,
    long nx,
    long ny,
    long nz,
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
    long xstride,
    long ystride,
    long zstride,
    double maxdist,
    double cosel0,
    double sinel0
    );

void atm_sim_compress_flag_extend_rank(
    uint8_t * hit,
    uint8_t * hit2,
    int ntask,
    int rank,
    long nx,
    long ny,
    long nz,
    long xstride,
    long ystride,
    long zstride
    );

double atm_sim_interp(
    double const & x,
    double const & y,
    double const & z,
    std::vector <long> & last_ind,
    std::vector <double> & last_nodes,
    double const & xstart,
    double const & ystart,
    double const & zstart,
    long const & nn,
    long const & nx,
    long const & ny,
    long const & nz,
    long const & xstride,
    long const & ystride,
    long const & zstride,
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
    long const & nelem,
    long const * compressed_index,
    long const * full_index,
    double const * realization
    );

int atm_sim_observe(
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
    long nn,
    long nx,
    long ny,
    long nz,
    long xstride,
    long ystride,
    long zstride,
    long nelem,
    long * compressed_index,
    long * full_index,
    double * realization
    );

void atm_sim_kolmogorov_init_rank(
    long nr,
    double * kolmo_x,
    double * kolmo_y,
    double rmin_kolmo,
    double rmax_kolmo,
    double rstep,
    double lmin,
    double lmax,
    int ntask,
    int rank
    );

double atm_sim_kolmogorov(
    double const & r,
    long const & nr,
    double const & rmin_kolmo,
    double const & rmax_kolmo,
    double const * kolmo_x,
    double const * kolmo_y
    );

double atm_sim_cov_eval(
    long const & nr,
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
    );

void atm_sim_ind2coord(
    double const & xstart,
    double const & ystart,
    double const & zstart,
    double const & xstep,
    double const & ystep,
    double const & zstep,
    long const & xstride,
    long const & ystride,
    long const & zstride,
    double const & xstrideinv,
    double const & ystrideinv,
    double const & zstrideinv,
    double const & cosel0,
    double const & sinel0,
    long const * full_index,
    long const & i,
    double * coord
    );

long atm_sim_coord2ind(
    double const & xstart,
    double const & ystart,
    double const & zstart,
    long const & xstride,
    long const & ystride,
    long const & zstride,
    double const & xstepinv,
    double const & ystepinv,
    double const & zstepinv,
    long const & nx,
    long const & ny,
    long const & nz,
    long const * compressed_index,
    double const & x,
    double const & y,
    double const & z
    );

cholmod_sparse * atm_sim_build_sparse_covariance(
    long ind_start,
    long ind_stop,
    long nr,
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
    long xstride,
    long ystride,
    long zstride,
    double z0,
    double cosel0,
    double sinel0,
    long const * full_index,
    bool smooth,
    double xxstep,
    double zzstep,
    int rank
    );

cholmod_sparse * atm_sim_sqrt_sparse_covariance(
    cholmod_sparse * cov,
    long ind_start,
    long ind_stop,
    int rank
    );

void atm_sim_apply_sparse_covariance(
    cholmod_sparse * sqrt_cov,
    long ind_start,
    long ind_stop,
    uint64_t key1,
    uint64_t key2,
    uint64_t counter1,
    uint64_t counter2,
    double * realization,
    int rank
    );

void atm_sim_compute_slice(
    long ind_start,
    long ind_stop,
    long nr,
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
    long xstride,
    long ystride,
    long zstride,
    double z0,
    double cosel0,
    double sinel0,
    long const * full_index,
    bool smooth,
    double xxstep,
    double zzstep,
    int rank,
    uint64_t key1,
    uint64_t key2,
    uint64_t counter1,
    uint64_t counter2,
    double * realization
    );
}

#endif // ifdef HAVE_CHOLMOD

#endif // ifndef TOAST_ATM_HPP
