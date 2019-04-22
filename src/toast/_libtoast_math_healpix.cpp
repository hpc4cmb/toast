
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>


void init_math_healpix(py::module & m) {
    m.def(
        "healpix_ang2vec", [](py::buffer theta, py::buffer phi,
                              py::buffer vec) {
            pybuffer_check_1D <double> (theta);
            pybuffer_check_1D <double> (phi);
            pybuffer_check_1D <double> (vec);
            py::buffer_info info_theta = theta.request();
            py::buffer_info info_phi = phi.request();
            py::buffer_info info_vec = vec.request();
            size_t nvec = (size_t)(info_vec.size / 3);
            if ((info_theta.size != info_phi.size) ||
                (info_theta.size != nvec)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * rawtheta = reinterpret_cast <double *> (info_theta.ptr);
            double * rawphi = reinterpret_cast <double *> (info_phi.ptr);
            double * rawvec = reinterpret_cast <double *> (info_vec.ptr);
            toast::healpix_ang2vec(info_theta.size, rawtheta, rawphi, rawvec);
            return;
        }, py::arg("theta"), py::arg("phi"), py::arg(
            "vec"), R"(
        Convert spherical coordinates to a unit vector.

        The theta angle is measured down from the North pole and phi is
        measured from the prime meridian.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            theta (array_like): The spherical coordinate theta angles in
                radians.
            phi (array like): The spherical coordinate phi angles in radians.
            vec (array_like):  The array of output vectors.

        Returns:
            None.

    )");

    m.def(
        "healpix_vec2ang", [](py::buffer vec, py::buffer theta,
                              py::buffer phi) {
            pybuffer_check_1D <double> (theta);
            pybuffer_check_1D <double> (phi);
            pybuffer_check_1D <double> (vec);
            py::buffer_info info_theta = theta.request();
            py::buffer_info info_phi = phi.request();
            py::buffer_info info_vec = vec.request();
            size_t nvec = (size_t)(info_vec.size / 3);
            if ((info_theta.size != info_phi.size) ||
                (info_theta.size != nvec)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * rawtheta = reinterpret_cast <double *> (info_theta.ptr);
            double * rawphi = reinterpret_cast <double *> (info_phi.ptr);
            double * rawvec = reinterpret_cast <double *> (info_vec.ptr);
            toast::healpix_vec2ang(info_theta.size, rawvec, rawtheta, rawphi);
            return;
        }, py::arg("vec"), py::arg("theta"), py::arg(
            "phi"), R"(
        Convert spherical coordinates to a unit vector.

        The theta angle is measured down from the North pole and phi is
        measured from the prime meridian.

        The results are stored in the output buffers.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            vec (array_like):  The array of input vectors.
            theta (array_like): Output spherical coordinate theta angles in
                radians.
            phi (array like): Output spherical coordinate phi angles in
                radians.

        Returns:
            None.

    )");

    m.def(
        "healpix_vec2angpa", [](py::buffer vec, py::buffer theta,
                                py::buffer phi, py::buffer pa) {
            pybuffer_check_1D <double> (theta);
            pybuffer_check_1D <double> (phi);
            pybuffer_check_1D <double> (pa);
            pybuffer_check_1D <double> (vec);
            py::buffer_info info_theta = theta.request();
            py::buffer_info info_phi = phi.request();
            py::buffer_info info_pa = pa.request();
            py::buffer_info info_vec = vec.request();
            size_t nvec = (size_t)(info_vec.size / 3);
            if ((info_theta.size != info_phi.size) ||
                (info_theta.size != info_pa.size) ||
                (info_theta.size != nvec)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * rawtheta = reinterpret_cast <double *> (info_theta.ptr);
            double * rawphi = reinterpret_cast <double *> (info_phi.ptr);
            double * rawpa = reinterpret_cast <double *> (info_pa.ptr);
            double * rawvec = reinterpret_cast <double *> (info_vec.ptr);
            toast::healpix_vec2ang(info_theta.size, rawvec, rawtheta, rawphi,
                                   rawpa);
            return;
        }, py::arg("vec"), py::arg("theta"), py::arg("phi"), py::arg(
            "pa"), R"(
        Convert spherical coordinates to a unit vector.

        The theta angle is measured down from the North pole and phi is
        measured from the prime meridian.  The position angle is with respect
        to the local meridian at the point described by the theta / phi
        coordinates.

        The results are stored in the output buffers.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            vec (array_like):  The array of input vectors.
            theta (array_like): Output spherical coordinate theta angles in
                radians.
            phi (array like): Output spherical coordinate phi angles in
                radians.
            pa (array like): Output spherical coordinate position angles in
                radians.

        Returns:
            None.

    )");

    py::class_ <toast::HealpixPixels, toast::HealpixPixels::puniq> (
        m, "HealpixPixels",
        R"(
        Healpix conversions at a particular NSIDE value.

        This class stores helper values for computing healpix conversions at
        a given NSIDE resolution.

        Args:
            nside (int):  The NSIDE value to use.

        )")
    .def(py::init <> ())
    .def(py::init <int64_t> (), py::arg("nside"))
    .def("reset", &toast::HealpixPixels::reset, py::arg(
             "nside"), R"(
        Reset the NSIDE value used for conversions

        Args:
            nside (int):  The NSIDE value to use.

        Returns:
            None

    )");


    return;
}

//
//         void vec2zphi(int64_t n, double const * vec, double * phi,
//                       int * region, double * z, double * rtz) const;
//
//         void theta2z(int64_t n, double const * theta, int * region, double *
// z,
//                      double * rtz) const;
//
//         void zphi2nest(int64_t n, double const * phi, int const * region,
//                        double const * z, double const * rtz,
//                        int64_t * pix) const;
//
//         void zphi2ring(int64_t n, double const * phi, int const * region,
//                        double const * z, double const * rtz,
//                        int64_t * pix) const;
//
//         void ang2nest(int64_t n, double const * theta, double const * phi,
//                       int64_t * pix) const;
//
//         void ang2ring(int64_t n, double const * theta, double const * phi,
//                       int64_t * pix) const;
//
//         void vec2nest(int64_t n, double const * vec, int64_t * pix) const;
//
//         void vec2ring(int64_t n, double const * vec, int64_t * pix) const;
//
//         void ring2nest(int64_t n, int64_t const * ringpix,
//                        int64_t * nestpix) const;
//
//         void nest2ring(int64_t n, int64_t const * nestpix,
//                        int64_t * ringpix) const;
//
//         void degrade_ring(int factor, int64_t n, int64_t const * inpix,
//                           int64_t * outpix) const;
//
//         void degrade_nest(int factor, int64_t n, int64_t const * inpix,
//                           int64_t * outpix) const;
//
//         void upgrade_ring(int factor, int64_t n, int64_t const * inpix,
//                           int64_t * outpix) const;
//
//         void upgrade_nest(int factor, int64_t n, int64_t const * inpix,
//                           int64_t * outpix) const;
