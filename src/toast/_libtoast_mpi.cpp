
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast_mpi.hpp>

// This fixes compilation errors with OpenMPI 4
// (see https://github.com/hpc4cmb/toast/issues/298)
struct ompi_communicator_t{};

// Currently the only compiled code that uses MPI and needs to be bound to python is
// the atmosphere simulation code.  If the number of things increases, we should split
// this file into multiple files.

void init_mpi_atm(py::module & m) {
#ifdef HAVE_CHOLMOD
    py::class_ <toast::mpi_atm_sim, toast::mpi_atm_sim::puniq> (
        m, "AtmSimMPI",
        R"(
        Class representing a single atmosphere simulation.

        This simulation consists of a particular realization of a "slab" of the
        atmosphere that moves with a constant wind speed and can be observed by
        individual detectors.

        Args:
            azmin (float):  The minimum of the azimuth range.
            azmax (float):  The maximum of the azimuth range.
            elmin (float):  The minimum of the elevation range.
            elmax (float):  The maximum of the elevation range.
            tmin (float):  The minimum of the time range.
            tmax (float):  The maximum of the time range.
            lmin_center (float):  Center point of the distribution of the dissipation
                scale of the Kolmogorov turbulence.
            lmin_sigma (float):  Width of the distribution of the dissipation
                scale of the Kolmogorov turbulence.
            lmax_center (float):  Center point of the distribution of the injection
                scale of the Kolmogorov turbulence.
            lmax_sigma (float):  Width of the distribution of the injection
                scale of the Kolmogorov turbulence.
            w_center (float):  Center point of the distribution of wind speed (m/s).
            w_sigma (float):  Width of the distribution of wind speed.
            wdir_center (float):  Center point of the distribution of wind direction
                (radians).
            wdir_sigma (float):  Width of the distribution of wind direction.
            z0_center (float):  Center point of the distribution of the water vapor (m).
            z0_sigma (float):  Width of the distribution of the water vapor.
            T0_center (float):  Center point of the distribution of ground temperature
                (Kelvin).
            T0_sigma (float):  Width of the distribution of ground temperature.
            zatm (float):  Atmosphere extent for temperature profile.
            zmax (float):  Water vaport extent for integration.
            xstep (float):  Size of volume element in the X direction.
            ystep (float):  Size of volume element in the Y direction.
            zstep (float):  Size of volume element in the Z direction.
            nelem_sim_max (int):  Size of the simulation slices.
            verbosity (int):  Controls logging.
            comm (mpi4py.MPI.Comm):  The MPI communicator.
            key1 (uint64):  Streamed RNG key 1.
            key2 (uint64):  Streamed RNG key 2.
            counterval1 (uint64):  Streamed RNG counter 1.
            counterval2 (uint64):  Streamed RNG counter 2.
            cachedir (str):  The location of the cached simulation.
            rmin (float):  Minimum line of sight observing distance.
            rmax (float):  Maximum line of sight observing distance.

        )")
    .def(py::init <double, double, double, double, double, double, double, double,
                   double, double, double, double, double, double, double, double,
                   double, double,
                   double, double, double, double, double, long, int, MPI_Comm,
                   uint64_t, uint64_t, uint64_t,
                   uint64_t, std::string, double, double> (), py::arg("azmin"), py::arg(
             "azmax"), py::arg("elmin"), py::arg("elmax"), py::arg("tmin"), py::arg(
             "tmax"), py::arg("lmin_center"), py::arg("lmin_sigma"), py::arg(
             "lmax_center"), py::arg("lmax_sigma"), py::arg("w_center"), py::arg(
             "w_sigma"), py::arg("wdir_center"), py::arg("wdir_sigma"), py::arg(
             "z0_center"), py::arg("z0_sigma"), py::arg("T0_center"), py::arg(
             "T0_sigma"), py::arg("zatm"), py::arg("zmax"), py::arg("xstep"), py::arg(
             "ystep"), py::arg("zstep"), py::arg("nelem_sim_max"), py::arg(
             "verbosity"), py::arg("comm"), py::arg("key1"), py::arg("key2"),
         py::arg("counterval1"), py::arg("counterval2"), py::arg("cachedir"), py::arg(
             "rmin"), py::arg("rmax")
         )
    .def("simulate", &toast::mpi_atm_sim::simulate, py::arg(
             "use_cache"), R"(
        Perform the simulation.

        Args:
            use_cache (bool):  If True, use the disk cache for save / load.

        Returns:
            (int):  A status value (zero == good).

    )")
    .def("observe", [](toast::mpi_atm_sim & self, py::buffer times,
                       py::buffer az, py::buffer el, py::buffer tod, double fixed_r) {
             pybuffer_check_1D <double> (times);
             pybuffer_check_1D <double> (az);
             pybuffer_check_1D <double> (el);
             pybuffer_check_1D <double> (tod);
             py::buffer_info info_times = times.request();
             py::buffer_info info_az = az.request();
             py::buffer_info info_el = el.request();
             py::buffer_info info_tod = tod.request();
             size_t nsamp = info_times.size;
             if ((info_az.size != nsamp) ||
                 (info_el.size != nsamp) ||
                 (info_tod.size != nsamp)) {
                 auto log = toast::Logger::get();
                 std::ostringstream o;
                 o << "Buffer sizes are not consistent.";
                 log.error(o.str().c_str());
                 throw std::runtime_error(o.str().c_str());
             }
             double * rawtimes = reinterpret_cast <double *> (info_times.ptr);
             double * rawaz = reinterpret_cast <double *> (info_az.ptr);
             double * rawel = reinterpret_cast <double *> (info_el.ptr);
             double * rawtod = reinterpret_cast <double *> (info_tod.ptr);
             auto status = self.observe(rawtimes, rawaz, rawel, rawtod, nsamp, fixed_r);
             return status;
         }, py::arg("times"), py::arg("az"), py::arg("el"), py::arg("tod"), py::arg(
             "fixed_r") = -1.0, R"(
            Observe the atmosphere with a detector.

            The timestamps and Azimuth / Elevation pointing are provided.  The TOD
            buffer is filled with the integrated atmospheric signal.

            For each sample, integrate along the line of sight by summing the
            atmosphere values. See Church (1995) Section 2.2, first equation.
            We omit the optical depth factor which is close to unity.

            Args:
                times (array_like):  Detector timestamps.
                az (array like):  Azimuth values.
                el (array_like):  Elevation values.
                tod (array_like):  The output buffer to fill.
                fixed_r (float):  If greater than zero, use this single radial value.

            Returns:
                (int):  A status value (zero == good).

        )")
    .def("__repr__",
         [](toast::mpi_atm_sim const & self) {
             std::ostringstream o;
             o << "<toast.AtmSimMPI";
             self.print(o);
             o << ">";
             return o.str();
         });
#endif // ifdef HAVE_CHOLMOD
    return;
}

PYBIND11_MODULE(_libtoast_mpi, m) {
    m.doc() = R"(
    Interface to C++ TOAST MPI library.

    )";

    init_mpi_atm(m);
}
