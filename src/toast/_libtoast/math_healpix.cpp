
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>


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
        Convert unit vectors to spherical coordinates.

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
        "healpix_vecs2angpa", [](py::buffer vec, py::buffer theta,
                                 py::buffer phi, py::buffer pa) {
            pybuffer_check_1D <double> (theta);
            pybuffer_check_1D <double> (phi);
            pybuffer_check_1D <double> (pa);
            pybuffer_check_1D <double> (vec);
            py::buffer_info info_theta = theta.request();
            py::buffer_info info_phi = phi.request();
            py::buffer_info info_pa = pa.request();
            py::buffer_info info_vec = vec.request();
            size_t nvec = (size_t)(info_vec.size / 6);
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
            toast::healpix_vecs2angpa(info_theta.size, rawvec, rawtheta, rawphi, rawpa);
            return;
        }, py::arg("vec"), py::arg("theta"), py::arg("phi"), py::arg(
            "pa"), R"(
        Convert direction / orientation unit vectors.

        The inputs are flat-packed pairs of direction and orientation unit
        vectors (6 float64 values total per sample).  The outputs are the
        theta, phi, and position angle of the location on the sphere.

        The theta angle is measured down from the North pole and phi is
        measured from the prime meridian.  The position angle is with respect
        to the local meridian at the point described by the theta / phi
        coordinates.

        The results are stored in the output buffers.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            vec (array_like):  The array of packed input direction and
                orientation vectors.
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

    )")
    .def("ang2nest", [](toast::HealpixPixels & self, py::buffer theta,
                        py::buffer phi, py::buffer pix) {
             pybuffer_check_1D <double> (theta);
             pybuffer_check_1D <double> (phi);
             pybuffer_check_1D <int64_t> (pix);
             py::buffer_info info_theta = theta.request();
             py::buffer_info info_phi = phi.request();
             py::buffer_info info_pix = pix.request();
             if ((info_theta.size != info_phi.size) ||
                 (info_theta.size != info_pix.size)) {
                 auto log = toast::Logger::get();
                 std::ostringstream o;
                 o << "Buffer sizes are not consistent.";
                 log.error(o.str().c_str());
                 throw std::runtime_error(o.str().c_str());
             }
             double * rawtheta = reinterpret_cast <double *> (info_theta.ptr);
             double * rawphi = reinterpret_cast <double *> (info_phi.ptr);
             int64_t * rawpix = reinterpret_cast <int64_t *> (info_pix.ptr);
             self.ang2nest(info_theta.size, rawtheta, rawphi, rawpix);
             return;
         }, py::arg("theta"), py::arg("phi"), py::arg(
             "pix"), R"(
            Convert spherical coordinates to pixels in NESTED ordering.

            The theta angle is measured down from the North pole and phi is
            measured from the prime meridian.

            The results are stored in the output buffers.  To guarantee SIMD
            vectorization, the input and output arrays should be aligned
            (i.e. use AlignedF64 / AlignedI64).

            Args:
                theta (array_like): Input spherical coordinate theta angles in
                    radians.
                phi (array like): Input spherical coordinate phi angles in
                    radians.
                pix (array like): Output pixel indices.

            Returns:
                None.

        )")
    .def("ang2ring", [](toast::HealpixPixels & self, py::buffer theta,
                        py::buffer phi, py::buffer pix) {
             pybuffer_check_1D <double> (theta);
             pybuffer_check_1D <double> (phi);
             pybuffer_check_1D <int64_t> (pix);
             py::buffer_info info_theta = theta.request();
             py::buffer_info info_phi = phi.request();
             py::buffer_info info_pix = pix.request();
             if ((info_theta.size != info_phi.size) ||
                 (info_theta.size != info_pix.size)) {
                 auto log = toast::Logger::get();
                 std::ostringstream o;
                 o << "Buffer sizes are not consistent.";
                 log.error(o.str().c_str());
                 throw std::runtime_error(o.str().c_str());
             }
             double * rawtheta = reinterpret_cast <double *> (info_theta.ptr);
             double * rawphi = reinterpret_cast <double *> (info_phi.ptr);
             int64_t * rawpix = reinterpret_cast <int64_t *> (info_pix.ptr);
             self.ang2ring(info_theta.size, rawtheta, rawphi, rawpix);
             return;
         }, py::arg("theta"), py::arg("phi"), py::arg(
             "pix"), R"(
            Convert spherical coordinates to pixels in RING ordering.

            The theta angle is measured down from the North pole and phi is
            measured from the prime meridian.

            The results are stored in the output buffers.  To guarantee SIMD
            vectorization, the input and output arrays should be aligned
            (i.e. use AlignedF64 / AlignedI64).

            Args:
                theta (array_like): Input spherical coordinate theta angles in
                    radians.
                phi (array like): Input spherical coordinate phi angles in
                    radians.
                pix (array like): Output pixel indices.

            Returns:
                None.

        )")
    .def("vec2nest", [](toast::HealpixPixels & self, py::buffer vec,
                        py::buffer pix) {
             pybuffer_check_1D <double> (vec);
             pybuffer_check_1D <int64_t> (pix);
             py::buffer_info info_vec = vec.request();
             py::buffer_info info_pix = pix.request();
             size_t nvec = (size_t)(info_vec.size / 3);
             if (nvec != info_pix.size) {
                 auto log = toast::Logger::get();
                 std::ostringstream o;
                 o << "Buffer sizes are not consistent.";
                 log.error(o.str().c_str());
                 throw std::runtime_error(o.str().c_str());
             }
             double * rawvec = reinterpret_cast <double *> (info_vec.ptr);
             int64_t * rawpix = reinterpret_cast <int64_t *> (info_pix.ptr);
             self.vec2nest(nvec, rawvec, rawpix);
             return;
         }, py::arg("vec"), py::arg(
             "pix"), R"(
            Convert unit vectors to pixels in NESTED ordering.

            The theta angle is measured down from the North pole and phi is
            measured from the prime meridian.

            The results are stored in the output buffer.  To guarantee SIMD
            vectorization, the input and output arrays should be aligned
            (i.e. use AlignedF64 / AlignedI64).

            Args:
                vec (array_like): Input packed unit vectors.
                pix (array like): Output pixel indices.

            Returns:
                None.

        )")
    .def("vec2ring", [](toast::HealpixPixels & self, py::buffer vec,
                        py::buffer pix) {
             pybuffer_check_1D <double> (vec);
             pybuffer_check_1D <int64_t> (pix);
             py::buffer_info info_vec = vec.request();
             py::buffer_info info_pix = pix.request();
             size_t nvec = (size_t)(info_vec.size / 3);
             if (nvec != info_pix.size) {
                 auto log = toast::Logger::get();
                 std::ostringstream o;
                 o << "Buffer sizes are not consistent.";
                 log.error(o.str().c_str());
                 throw std::runtime_error(o.str().c_str());
             }
             double * rawvec = reinterpret_cast <double *> (info_vec.ptr);
             int64_t * rawpix = reinterpret_cast <int64_t *> (info_pix.ptr);
             self.vec2ring(nvec, rawvec, rawpix);
             return;
         }, py::arg("vec"), py::arg(
             "pix"), R"(
            Convert unit vectors to pixels in RING ordering.

            The theta angle is measured down from the North pole and phi is
            measured from the prime meridian.

            The results are stored in the output buffer.  To guarantee SIMD
            vectorization, the input and output arrays should be aligned
            (i.e. use AlignedF64 / AlignedI64).

            Args:
                vec (array_like): Input packed unit vectors.
                pix (array like): Output pixel indices.

            Returns:
                None.

        )")
    .def("ring2nest", [](toast::HealpixPixels & self, py::buffer in,
                         py::buffer out) {
             pybuffer_check_1D <int64_t> (in);
             pybuffer_check_1D <int64_t> (out);
             py::buffer_info info_in = in.request();
             py::buffer_info info_out = out.request();
             if (info_in.size != info_out.size) {
                 auto log = toast::Logger::get();
                 std::ostringstream o;
                 o << "Buffer sizes are not consistent.";
                 log.error(o.str().c_str());
                 throw std::runtime_error(o.str().c_str());
             }
             int64_t * rawin = reinterpret_cast <int64_t *> (info_in.ptr);
             int64_t * rawout = reinterpret_cast <int64_t *> (info_out.ptr);
             self.ring2nest(info_in.size, rawin, rawout);
             return;
         }, py::arg("in"), py::arg(
             "out"), R"(
            Convert RING ordered pixel numbers into NESTED ordering.

            The results are stored in the output buffer.  To guarantee SIMD
            vectorization, the input and output arrays should be aligned
            (i.e. use AlignedF64 / AlignedI64).

            Args:
                in (array_like): Input pixel indices.
                out (array like): Output pixel indices.

            Returns:
                None.

        )")
    .def("nest2ring", [](toast::HealpixPixels & self, py::buffer in,
                         py::buffer out) {
             pybuffer_check_1D <int64_t> (in);
             pybuffer_check_1D <int64_t> (out);
             py::buffer_info info_in = in.request();
             py::buffer_info info_out = out.request();
             if (info_in.size != info_out.size) {
                 auto log = toast::Logger::get();
                 std::ostringstream o;
                 o << "Buffer sizes are not consistent.";
                 log.error(o.str().c_str());
                 throw std::runtime_error(o.str().c_str());
             }
             int64_t * rawin = reinterpret_cast <int64_t *> (info_in.ptr);
             int64_t * rawout = reinterpret_cast <int64_t *> (info_out.ptr);
             self.nest2ring(info_in.size, rawin, rawout);
             return;
         }, py::arg("in"), py::arg(
             "out"), R"(
            Convert NESTED ordered pixel numbers into RING ordering.

            The results are stored in the output buffer.  To guarantee SIMD
            vectorization, the input and output arrays should be aligned
            (i.e. use AlignedF64 / AlignedI64).

            Args:
                in (array_like): Input pixel indices.
                out (array like): Output pixel indices.

            Returns:
                None.

        )")
    .def("degrade_ring", [](toast::HealpixPixels & self, int factor,
                            py::buffer in, py::buffer out) {
             pybuffer_check_1D <int64_t> (in);
             pybuffer_check_1D <int64_t> (out);
             py::buffer_info info_in = in.request();
             py::buffer_info info_out = out.request();
             if (info_in.size != info_out.size) {
                 auto log = toast::Logger::get();
                 std::ostringstream o;
                 o << "Buffer sizes are not consistent.";
                 log.error(o.str().c_str());
                 throw std::runtime_error(o.str().c_str());
             }
             int64_t * rawin = reinterpret_cast <int64_t *> (info_in.ptr);
             int64_t * rawout = reinterpret_cast <int64_t *> (info_out.ptr);
             self.degrade_ring(info_in.size, factor, rawin, rawout);
             return;
         }, py::arg("factor"), py::arg("in"), py::arg(
             "out"), R"(
            Degrade RING ordered pixel numbers.

            Each 'factor' is one division by two in the NSIDE resolution.  So
            a factor of '3' would divide the NSIDE value by 8.

            The results are stored in the output buffer.  To guarantee SIMD
            vectorization, the input and output arrays should be aligned
            (i.e. use AlignedF64 / AlignedI64).

            Args:
                factor (int):  The degrade factor.
                in (array_like): Input pixel indices.
                out (array like): Output pixel indices.

            Returns:
                None.

        )")
    .def("degrade_nest", [](toast::HealpixPixels & self, int factor,
                            py::buffer in, py::buffer out) {
             pybuffer_check_1D <int64_t> (in);
             pybuffer_check_1D <int64_t> (out);
             py::buffer_info info_in = in.request();
             py::buffer_info info_out = out.request();
             if (info_in.size != info_out.size) {
                 auto log = toast::Logger::get();
                 std::ostringstream o;
                 o << "Buffer sizes are not consistent.";
                 log.error(o.str().c_str());
                 throw std::runtime_error(o.str().c_str());
             }
             int64_t * rawin = reinterpret_cast <int64_t *> (info_in.ptr);
             int64_t * rawout = reinterpret_cast <int64_t *> (info_out.ptr);
             self.degrade_nest(info_in.size, factor, rawin, rawout);
             return;
         }, py::arg("factor"), py::arg("in"), py::arg(
             "out"), R"(
            Degrade NESTED ordered pixel numbers.

            Each 'factor' is one division by two in the NSIDE resolution.  So
            a factor of '3' would divide the NSIDE value by 8.

            The results are stored in the output buffer.  To guarantee SIMD
            vectorization, the input and output arrays should be aligned
            (i.e. use AlignedF64 / AlignedI64).

            Args:
                factor (int):  The degrade factor.
                in (array_like): Input pixel indices.
                out (array like): Output pixel indices.

            Returns:
                None.

        )")
    .def("upgrade_ring", [](toast::HealpixPixels & self, int factor,
                            py::buffer in, py::buffer out) {
             pybuffer_check_1D <int64_t> (in);
             pybuffer_check_1D <int64_t> (out);
             py::buffer_info info_in = in.request();
             py::buffer_info info_out = out.request();
             if (info_in.size != info_out.size) {
                 auto log = toast::Logger::get();
                 std::ostringstream o;
                 o << "Buffer sizes are not consistent.";
                 log.error(o.str().c_str());
                 throw std::runtime_error(o.str().c_str());
             }
             int64_t * rawin = reinterpret_cast <int64_t *> (info_in.ptr);
             int64_t * rawout = reinterpret_cast <int64_t *> (info_out.ptr);
             self.upgrade_ring(info_in.size, factor, rawin, rawout);
             return;
         }, py::arg("factor"), py::arg("in"), py::arg(
             "out"), R"(
            Upgrade RING ordered pixel numbers.

            Each 'factor' is one multiplication by two in the NSIDE
            resolution.  So a factor of '3' would multiply the NSIDE value
            by 8.

            The results are stored in the output buffer.  To guarantee SIMD
            vectorization, the input and output arrays should be aligned
            (i.e. use AlignedF64 / AlignedI64).

            Args:
                factor (int):  The upgrade factor.
                in (array_like): Input pixel indices.
                out (array like): Output pixel indices.

            Returns:
                None.

        )")
    .def("upgrade_nest", [](toast::HealpixPixels & self, int factor,
                            py::buffer in, py::buffer out) {
             pybuffer_check_1D <int64_t> (in);
             pybuffer_check_1D <int64_t> (out);
             py::buffer_info info_in = in.request();
             py::buffer_info info_out = out.request();
             if (info_in.size != info_out.size) {
                 auto log = toast::Logger::get();
                 std::ostringstream o;
                 o << "Buffer sizes are not consistent.";
                 log.error(o.str().c_str());
                 throw std::runtime_error(o.str().c_str());
             }
             int64_t * rawin = reinterpret_cast <int64_t *> (info_in.ptr);
             int64_t * rawout = reinterpret_cast <int64_t *> (info_out.ptr);
             self.degrade_nest(info_in.size, factor, rawin, rawout);
             return;
         }, py::arg("factor"), py::arg("in"), py::arg(
             "out"), R"(
            Upgrade NESTED ordered pixel numbers.

            Each 'factor' is one multiplication by two in the NSIDE
            resolution.  So a factor of '3' would multiply the NSIDE value
            by 8.

            The results are stored in the output buffer.  To guarantee SIMD
            vectorization, the input and output arrays should be aligned
            (i.e. use AlignedF64 / AlignedI64).

            Args:
                factor (int):  The upgrade factor.
                in (array_like): Input pixel indices.
                out (array like): Output pixel indices.

            Returns:
                None.

        )");


    return;
}
