
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>


void init_atm(py::module & m) {
    #ifdef HAVE_AATM
    m.def("atm_absorption_coefficient", &toast::atm_get_absorption_coefficient,
          py::arg("altitude"), py::arg("temperature"), py::arg("pressure"),
          py::arg("pwv"), py::arg(
              "freq"), R"(
            Compute the absorption coefficient.

            Args:
                altitude (float):  The observing altitude in meters.
                temperature (float):  The observing temperature in Kelvin.
                pressure (float):  The observing pressure in Pascals.
                pwv (float):  The precipitable water vapor in mm.
                freq (float):  Observing frequency in GHz.

            Returns:
                (float):  The absorption coefficient.

    )");

    m.def("atm_atmospheric_loading", &toast::atm_get_atmospheric_loading,
          py::arg("altitude"), py::arg("temperature"), py::arg("pressure"),
          py::arg("pwv"), py::arg(
              "freq"), R"(
            Return the equivalent blackbody temperature in Kelvin.

            Args:
                altitude (float):  The observing altitude in meters.
                temperature (float):  The observing temperature in Kelvin.
                pressure (float):  The observing pressure in Pascals.
                pwv (float):  The precipitable water vapor in mm.
                freq (float):  Observing frequency in GHz.

            Returns:
                (float):  The temperature.

    )");

    m.def("atm_absorption_coefficient_vec", [](double altitude, double temperature,
                                               double pressure, double pwv,
                                               double freqmin, double freqmax,
                                               size_t nfreq) {
              py::array_t <double> ret;
              ret.resize({nfreq});
              py::buffer_info info = ret.request();
              double * raw = static_cast <double *> (info.ptr);
              auto blah = toast::atm_get_absorption_coefficient_vec(
                  altitude, temperature, pressure, pwv, freqmin, freqmax, nfreq, raw);
              return ret;
          }, py::arg("altitude"), py::arg("temperature"), py::arg("pressure"),
          py::arg("pwv"), py::arg("freqmin"), py::arg("freqmax"), py::arg(
              "nfreq"), R"(
            Compute a vector of absorption coefficients.

            Args:
                altitude (float):  The observing altitude in meters.
                temperature (float):  The observing temperature in Kelvin.
                pressure (float):  The observing pressure in Pascals.
                pwv (float):  The precipitable water vapor in mm.
                freqmin (float):  Minimum observing frequency in GHz.
                freqmax (float):  Maximum observing frequency in GHz.
                nfreq (int):  Number of frequency points to compute.

            Returns:
                (array):  The absorption coefficients at the specified frequencies.

    )");

    m.def("atm_atmospheric_loading_vec", [](double altitude, double temperature,
                                            double pressure, double pwv, double freqmin,
                                            double freqmax, size_t nfreq) {
              py::array_t <double> ret;
              ret.resize({nfreq});
              py::buffer_info info = ret.request();
              double * raw = static_cast <double *> (info.ptr);
              auto blah = toast::atm_get_atmospheric_loading_vec(
                  altitude, temperature, pressure, pwv, freqmin, freqmax, nfreq, raw);
              return ret;
          }, py::arg("altitude"), py::arg("temperature"), py::arg("pressure"),
          py::arg("pwv"), py::arg("freqmin"), py::arg("freqmax"), py::arg(
              "nfreq"), R"(
            Compute a vector of equivalent blackbody temperatures.

            Args:
                altitude (float):  The observing altitude in meters.
                temperature (float):  The observing temperature in Kelvin.
                pressure (float):  The observing pressure in Pascals.
                pwv (float):  The precipitable water vapor in mm.
                freqmin (float):  Minimum observing frequency in GHz.
                freqmax (float):  Maximum observing frequency in GHz.
                nfreq (int):  Number of frequency points to compute.

            Returns:
                (array):  The temperatures at the specified frequencies.

    )");
    #endif // ifdef HAVE_AATM

    #ifdef HAVE_CHOLMOD

    m.def("atm_sim_compute_slice",
          [](int64_t ind_start,
             int64_t ind_stop,
             double rmin_kolmo,
             double rmax_kolmo,
             py::buffer kolmo_x,
             py::buffer kolmo_y,
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
             py::buffer full_index,
             bool smooth,
             double xxstep,
             double zzstep,
             int rank,
             uint64_t key1,
             uint64_t key2,
             uint64_t counter1,
             uint64_t counter2,
             py::buffer realization) {
              pybuffer_check_1D <double> (kolmo_x);
              pybuffer_check_1D <double> (kolmo_y);
              pybuffer_check_1D <int64_t> (full_index);
              pybuffer_check_1D <double> (realization);
              py::buffer_info info_kolmo_x = kolmo_x.request();
              py::buffer_info info_kolmo_y = kolmo_y.request();
              py::buffer_info info_full_index = full_index.request();
              py::buffer_info info_realization = realization.request();
              int64_t nr = info_kolmo_x.size;
              if (info_kolmo_y.size != nr) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "kolmo_x / kolmo_y sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              int64_t nelem = info_realization.size;
              if (info_full_index.size != nelem) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "full_index and realization sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              double * raw_kolmo_x = reinterpret_cast <double *> (info_kolmo_x.ptr);
              double * raw_kolmo_y = reinterpret_cast <double *> (info_kolmo_y.ptr);
              double * raw_realiz = reinterpret_cast <double *> (info_realization.ptr);
              int64_t * raw_full = reinterpret_cast <int64_t *> (info_full_index.ptr);
              toast::atm_sim_compute_slice(
                  ind_start,
                  ind_stop,
                  nr,
                  rmin_kolmo,
                  rmax_kolmo,
                  raw_kolmo_x,
                  raw_kolmo_y,
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
                  raw_full,
                  smooth,
                  xxstep,
                  zzstep,
                  rank,
                  key1,
                  key2,
                  counter1,
                  counter2,
                  raw_realiz
              );
              return;
          }, R"(
     Internal function used by AtmSim class.
    )");

    m.def("atm_sim_observe",
          [](
              py::buffer times,
              py::buffer az,
              py::buffer el,
              py::buffer tod,
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
              int64_t nx,
              int64_t ny,
              int64_t nz,
              int64_t xstride,
              int64_t ystride,
              int64_t zstride,
              py::buffer compressed_index,
              py::buffer full_index,
              py::buffer realization
          ) {
              pybuffer_check_1D <double> (times);
              pybuffer_check_1D <double> (az);
              pybuffer_check_1D <double> (el);
              pybuffer_check_1D <double> (tod);
              pybuffer_check_1D <int64_t> (compressed_index);
              pybuffer_check_1D <int64_t> (full_index);
              pybuffer_check_1D <double> (realization);
              py::buffer_info info_times = times.request();
              py::buffer_info info_az = az.request();
              py::buffer_info info_el = el.request();
              py::buffer_info info_tod = tod.request();
              py::buffer_info info_comp_index = compressed_index.request();
              py::buffer_info info_full_index = full_index.request();
              py::buffer_info info_realiz = realization.request();
              int64_t nsamp = info_times.size;
              if ((info_az.size != nsamp) || (info_el.size != nsamp)
                  || (info_tod.size != nsamp)) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "time domain buffer sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              int64_t nelem = info_realiz.size;
              if (info_full_index.size != nelem) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "full_index and realization sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              int64_t nn = info_comp_index.size;
              double * raw_times = reinterpret_cast <double *> (info_times.ptr);
              double * raw_az = reinterpret_cast <double *> (info_az.ptr);
              double * raw_el = reinterpret_cast <double *> (info_el.ptr);
              double * raw_tod = reinterpret_cast <double *> (info_tod.ptr);
              double * raw_realiz = reinterpret_cast <double *> (info_realiz.ptr);
              int64_t * raw_full = reinterpret_cast <int64_t *> (info_full_index.ptr);
              int64_t * raw_comp = reinterpret_cast <int64_t *> (info_comp_index.ptr);
              int status = toast::atm_sim_observe(
                  nsamp,
                  raw_times,
                  raw_az,
                  raw_el,
                  raw_tod,
                  T0,
                  azmin,
                  azmax,
                  elmin,
                  elmax,
                  tmin,
                  tmax,
                  rmin,
                  rmax,
                  fixed_r,
                  zatm,
                  zmax,
                  wx,
                  wy,
                  wz,
                  xstep,
                  ystep,
                  zstep,
                  xstart,
                  delta_x,
                  ystart,
                  delta_y,
                  zstart,
                  delta_z,
                  maxdist,
                  nn,
                  nx,
                  ny,
                  nz,
                  xstride,
                  ystride,
                  zstride,
                  nelem,
                  raw_comp,
                  raw_full,
                  raw_realiz
              );
              return status;
          }, R"(
     Internal function used by AtmSim class.
    )");

    m.def("atm_sim_compress_flag_hits_rank",
          [](
              py::buffer hit,
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
              pybuffer_check_1D <uint8_t> (hit);
              py::buffer_info info_hit = hit.request();
              int64_t nn = info_hit.size;
              uint8_t * raw_hit = reinterpret_cast <uint8_t *> (info_hit.ptr);
              toast::atm_sim_compress_flag_hits_rank(
                  nn,
                  raw_hit,
                  ntask,
                  rank,
                  nx,
                  ny,
                  nz,
                  xstart,
                  ystart,
                  zstart,
                  delta_t,
                  delta_az,
                  elmin,
                  elmax,
                  wx,
                  wy,
                  wz,
                  xstep,
                  ystep,
                  zstep,
                  xstride,
                  ystride,
                  zstride,
                  maxdist,
                  cosel0,
                  sinel0
              );
              return;
          }, R"(
     Internal function used by AtmSim class.
    )");

    m.def("atm_sim_compress_flag_extend_rank",
          [](
              py::buffer hit,
              py::buffer hit2,
              int ntask,
              int rank,
              int64_t nx,
              int64_t ny,
              int64_t nz,
              int64_t xstride,
              int64_t ystride,
              int64_t zstride
          ) {
              pybuffer_check_1D <uint8_t> (hit);
              pybuffer_check_1D <uint8_t> (hit2);
              py::buffer_info info_hit = hit.request();
              py::buffer_info info_hit2 = hit2.request();
              int64_t nn = info_hit.size;
              if (info_hit2.size != nn) {
                  auto log = toast::Logger::get();
                  std::ostringstream o;
                  o << "hit and hit2 sizes are not consistent.";
                  log.error(o.str().c_str());
                  throw std::runtime_error(o.str().c_str());
              }
              uint8_t * raw_hit = reinterpret_cast <uint8_t *> (info_hit.ptr);
              uint8_t * raw_hit2 = reinterpret_cast <uint8_t *> (info_hit2.ptr);
              toast::atm_sim_compress_flag_extend_rank(
                  raw_hit,
                  raw_hit2,
                  ntask,
                  rank,
                  nx,
                  ny,
                  nz,
                  xstride,
                  ystride,
                  zstride
              );
              return;
          }, R"(
     Internal function used by AtmSim class.
    )");

    m.def("atm_sim_kolmogorov_init_rank",
          [](
              int64_t nr,
              double rmin_kolmo,
              double rmax_kolmo,
              double rstep,
              double lmin,
              double lmax,
              int ntask,
              int rank
          ) {
              py::array_t <double> kolmo_x;
              kolmo_x.resize({nr});
              py::array_t <double> kolmo_y;
              kolmo_y.resize({nr});
              py::buffer_info info_kolmo_x = kolmo_x.request();
              double * raw_kolmo_x = static_cast <double *> (info_kolmo_x.ptr);
              py::buffer_info info_kolmo_y = kolmo_y.request();
              double * raw_kolmo_y = static_cast <double *> (info_kolmo_y.ptr);
              toast::atm_sim_kolmogorov_init_rank(
                  nr,
                  raw_kolmo_x,
                  raw_kolmo_y,
                  rmin_kolmo,
                  rmax_kolmo,
                  rstep,
                  lmin,
                  lmax,
                  ntask,
                  rank
              );
              return py::make_tuple(kolmo_x, kolmo_y);
          }, R"(
       Internal function used by AtmSim class.
      )");

    #endif // ifdef HAVE_CHOLMOD

    return;
}
