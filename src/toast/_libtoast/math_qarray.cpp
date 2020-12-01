
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <_libtoast.hpp>


void init_math_qarray(py::module & m) {
    // Quaternion arrays

    m.def(
        "qa_inv", [](py::buffer data) {
            pybuffer_check_1D <double> (data);
            py::buffer_info info = data.request();
            double * raw = reinterpret_cast <double *> (info.ptr);
            size_t nquat = (size_t)(info.size / 4);
            toast::qa_inv(nquat, raw);
            return;
        }, py::arg(
            "data"), R"(
        Invert the array of quaternions.

        The operation is done in place.  To guarantee SIMD vectorization, the
        data buffer should be aligned (i.e. use an AlignedF64).

        Args:
            data (array_like):  flattened 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "qa_amplitude", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            size_t nquat = (size_t)(info_in.size / 4);
            if (info_out.size != nquat) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::qa_amplitude(nquat, 4, 4, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute the amplitude (norm) of the input quaternions.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  flattened 1D array of float64 values.
            out (array_like):  flattened 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "qa_normalize", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            size_t nquat = (size_t)(info_in.size / 4);
            if (info_out.size != info_in.size) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::qa_normalize(nquat, 4, 4, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Normalize a quaternion array.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  flattened 1D array of float64 values.
            out (array_like):  flattened 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "qa_normalize_inplace", [](py::buffer data) {
            pybuffer_check_1D <double> (data);
            py::buffer_info info = data.request();
            size_t nquat = (size_t)(info.size / 4);
            double * raw = reinterpret_cast <double *> (info.ptr);
            toast::qa_normalize_inplace(nquat, 4, 4, raw);
            return;
        }, py::arg(
            "data"), R"(
        Normalize a quaternion array in place.

        The data is modified in place.  To guarantee SIMD vectorization, the
        input array should be aligned (i.e. use an AlignedF64).

        Args:
            data (array_like):  flattened 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "qa_rotate", [](py::buffer q_in, py::buffer v_in, py::buffer v_out) {
            pybuffer_check_1D <double> (q_in);
            pybuffer_check_1D <double> (v_in);
            pybuffer_check_1D <double> (v_out);
            py::buffer_info info_qin = q_in.request();
            py::buffer_info info_vin = v_in.request();
            py::buffer_info info_vout = v_out.request();
            size_t nquat = (size_t)(info_qin.size / 4);
            size_t nvec = (size_t)(info_vin.size / 3);
            size_t nout = (size_t)(info_vout.size / 3);
            size_t check = (nquat > nvec) ? nquat : nvec;
            if (nout != check) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                o << " " << nquat << " " << nvec << " " << nout;
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * qinraw = reinterpret_cast <double *> (info_qin.ptr);
            double * vinraw = reinterpret_cast <double *> (info_vin.ptr);
            double * voutraw = reinterpret_cast <double *> (info_vout.ptr);
            toast::qa_rotate(nquat, qinraw, nvec, vinraw, voutraw);
            return;
        }, py::arg("q_in"), py::arg("v_in"), py::arg(
            "v_out"), R"(
        Rotate vectors with quaternions.

        The number of quaternions and vectors passed in should either be
        equal or there should be one quaternion (which is applied to all
        vectors) or one vector (which is rotated by all quaternions).

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            q_in (array_like):  flattened 1D array of float64 values.
            v_in (array_like):  flattened 1D array of float64 values.
            v_out (array_like):  flattened 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "qa_mult", [](py::buffer p_in, py::buffer q_in, py::buffer out) {
            pybuffer_check_1D <double> (p_in);
            pybuffer_check_1D <double> (q_in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_pin = p_in.request();
            py::buffer_info info_qin = q_in.request();
            py::buffer_info info_out = out.request();
            size_t np = (size_t)(info_pin.size / 4);
            size_t nq = (size_t)(info_qin.size / 4);
            size_t nout = (size_t)(info_out.size / 4);
            size_t check = (np > nq) ? np : nq;
            if (nout != check) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * pinraw = reinterpret_cast <double *> (info_pin.ptr);
            double * qinraw = reinterpret_cast <double *> (info_qin.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::qa_mult(np, pinraw, nq, qinraw, outraw);
            return;
        }, py::arg("p_in"), py::arg("q_in"), py::arg(
            "out"), R"(
        Multiply quaternion arrays.

        The number of quaternions in both input arrays should either be
        equal or there should be one quaternion in one of the arrays (which
        is then multiplied by all quaternions in the other array).

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            p_in (array_like):  flattened 1D array of float64 values.
            q_in (array_like):  flattened 1D array of float64 values.
            out (array_like):  flattened 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "qa_slerp", [](py::buffer time, py::buffer targettime,
                       py::buffer q_in, py::buffer q_out) {
            pybuffer_check_1D <double> (time);
            pybuffer_check_1D <double> (targettime);
            pybuffer_check_1D <double> (q_in);
            pybuffer_check_1D <double> (q_out);
            py::buffer_info info_time = time.request();
            py::buffer_info info_tgtime = targettime.request();
            py::buffer_info info_qin = q_in.request();
            py::buffer_info info_qout = q_out.request();
            size_t ntime = info_time.size;
            size_t ntgtime = info_tgtime.size;
            size_t nqin = (size_t)(info_qin.size / 4);
            size_t nqout = (size_t)(info_qout.size / 4);
            if ((ntime != nqin) || (ntgtime != nqout)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * timeraw = reinterpret_cast <double *> (info_time.ptr);
            double * tgtimeraw = reinterpret_cast <double *> (info_tgtime.ptr);
            double * qinraw = reinterpret_cast <double *> (info_qin.ptr);
            double * qoutraw = reinterpret_cast <double *> (info_qout.ptr);
            toast::qa_slerp(ntime, ntgtime, timeraw, tgtimeraw, qinraw,
                            qoutraw);
            return;
        }, py::arg("time"), py::arg("targettime"), py::arg("q_in"), py::arg(
            "q_out"), R"(
        Spherical Linear Interpolation of quaternions.

        For input quaternions at the given times, interpolate these to the
        target times.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            time (array_like):  time values.
            targettime (array_like): target time values.
            q_in (array_like):  flattened 1D array of float64 values.
            q_out (array_like):  flattened 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "qa_exp", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            size_t nin = (size_t)(info_in.size / 4);
            size_t nout = (size_t)(info_out.size / 4);
            if (nin != nout) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::qa_exp(nin, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute the exponential of a quaternion array.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  flattened 1D array of float64 values.
            out (array_like):  flattened 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "qa_ln", [](py::buffer in, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_out = out.request();
            size_t nin = (size_t)(info_in.size / 4);
            size_t nout = (size_t)(info_out.size / 4);
            if (nin != nout) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::qa_ln(nin, inraw, outraw);
            return;
        }, py::arg("in"), py::arg(
            "out"), R"(
        Compute the natural log of a quaternion array.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  flattened 1D array of float64 values.
            out (array_like):  flattened 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "qa_pow", [](py::buffer in, py::buffer pw, py::buffer out) {
            pybuffer_check_1D <double> (in);
            pybuffer_check_1D <double> (pw);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_in = in.request();
            py::buffer_info info_pw = pw.request();
            py::buffer_info info_out = out.request();
            size_t npw = info_pw.size;
            size_t nin = (size_t)(info_in.size / 4);
            size_t nout = (size_t)(info_out.size / 4);
            if ((nin != nout) || ((npw != nin) && (npw != 1))) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * pwraw = reinterpret_cast <double *> (info_pw.ptr);
            double * inraw = reinterpret_cast <double *> (info_in.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::qa_pow(nin, npw, pwraw, inraw, outraw);
            return;
        }, py::arg("in"), py::arg("pw"), py::arg(
            "out"), R"(
        Raise a quaternion array to a real power.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            in (array_like):  flattened 1D array of float64 values.
            pw (array_like):  array of powers.
            out (array_like):  flattened 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "qa_from_axisangle", [](py::buffer axis, py::buffer angle,
                                py::buffer out) {
            pybuffer_check_1D <double> (axis);
            pybuffer_check_1D <double> (angle);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_axis = axis.request();
            py::buffer_info info_angle = angle.request();
            py::buffer_info info_out = out.request();
            size_t nang = info_angle.size;
            size_t naxis = (size_t)(info_axis.size / 3);
            size_t check = (nang > naxis) ? nang : naxis;
            size_t nout = (size_t)(info_out.size / 4);
            if (((nang != naxis) && (nang != 1) && (naxis != 1))
                || (check != nout)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * angleraw = reinterpret_cast <double *> (info_angle.ptr);
            double * axisraw = reinterpret_cast <double *> (info_axis.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::qa_from_axisangle(naxis, axisraw, nang, angleraw, outraw);
            return;
        }, py::arg("axis"), py::arg("angle"), py::arg(
            "out"), R"(
        Create quaternions from axis / angle information.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            axis (array_like):  flattened 1D array of float64 values.
            angle (array_like):  array of angles.
            out (array_like):  flattened 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "qa_to_axisangle", [](py::buffer q_in, py::buffer axis,
                              py::buffer angle) {
            pybuffer_check_1D <double> (axis);
            pybuffer_check_1D <double> (angle);
            pybuffer_check_1D <double> (q_in);
            py::buffer_info info_axis = axis.request();
            py::buffer_info info_angle = angle.request();
            py::buffer_info info_qin = q_in.request();
            size_t nang = info_angle.size;
            size_t naxis = (size_t)(info_axis.size / 3);
            size_t nq = (size_t)(info_qin.size / 4);
            if ((nang != naxis) || (nang != nq)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * angleraw = reinterpret_cast <double *> (info_angle.ptr);
            double * axisraw = reinterpret_cast <double *> (info_axis.ptr);
            double * qinraw = reinterpret_cast <double *> (info_qin.ptr);
            toast::qa_to_axisangle(nang, qinraw, axisraw, angleraw);
            return;
        }, py::arg("q_in"), py::arg("axis"), py::arg(
            "angle"), R"(
        Convert quaternions into axis / angle representation.

        The results are stored in the output buffers.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            q_in (array_like):  flattened 1D array of float64 values.
            axis (array_like):  flattened 1D array of float64 values.
            angle (array_like):  array of angles.

        Returns:
            None

    )");

    m.def(
        "qa_to_rotmat", [](py::buffer q_in, py::buffer mat) {
            pybuffer_check_1D <double> (mat);
            pybuffer_check_1D <double> (q_in);
            py::buffer_info info_mat = mat.request();
            py::buffer_info info_qin = q_in.request();
            size_t nmat = (size_t)(info_mat.size / 9);
            size_t nq = (size_t)(info_qin.size / 4);
            if (nmat != nq) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * matraw = reinterpret_cast <double *> (info_mat.ptr);
            double * qinraw = reinterpret_cast <double *> (info_qin.ptr);
            for (size_t i = 0; i < nq; ++i) {
                toast::qa_to_rotmat(&(qinraw[4 * i]), &(matraw[9 * i]));
            }
            return;
        }, py::arg("q_in"), py::arg(
            "mat"), R"(
        Convert quaternions into rotation matrices.

        The results are stored in the output buffers.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            q_in (array_like):  flattened 1D array of float64 values.
            mat (array_like):  flattened 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "qa_from_rotmat", [](py::buffer mat, py::buffer q_out) {
            pybuffer_check_1D <double> (mat);
            pybuffer_check_1D <double> (q_out);
            py::buffer_info info_mat = mat.request();
            py::buffer_info info_qout = q_out.request();
            size_t nmat = (size_t)(info_mat.size / 9);
            size_t nq = (size_t)(info_qout.size / 4);
            if (nmat != nq) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * matraw = reinterpret_cast <double *> (info_mat.ptr);
            double * qoutraw = reinterpret_cast <double *> (info_qout.ptr);
            for (size_t i = 0; i < nq; ++i) {
                toast::qa_from_rotmat(&(matraw[9 * i]), &(qoutraw[4 * i]));
            }
            return;
        }, py::arg("mat"), py::arg(
            "q_out"), R"(
        Convert rotation matrices into quaternions.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            mat (array_like):  flattened 1D array of float64 values.
            q_out (array_like):  flattened 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "qa_from_vectors", [](py::buffer vec1, py::buffer vec2,
                              py::buffer out) {
            pybuffer_check_1D <double> (vec1);
            pybuffer_check_1D <double> (vec2);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_vec1 = vec1.request();
            py::buffer_info info_vec2 = vec2.request();
            py::buffer_info info_out = out.request();
            size_t nv1 = (size_t)(info_vec1.size / 3);
            size_t nv2 = (size_t)(info_vec2.size / 3);
            size_t nout = (size_t)(info_out.size / 4);
            if ((nv1 != nv2) || (nv1 != nout)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * v1raw = reinterpret_cast <double *> (info_vec1.ptr);
            double * v2raw = reinterpret_cast <double *> (info_vec2.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::qa_from_vectors(nv1, v1raw, v2raw, outraw);
            return;
        }, py::arg("vec1"), py::arg("vec2"), py::arg(
            "out"), R"(
        Create quaternions from two vectors.

        Each output quaternion describes the rotation from vec1 to vec2.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            vec1 (array_like):  flattened 1D array of float64 values.
            vec2 (array_like):  flattened 1D array of float64 values.
            out (array_like):  flattened 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "qa_from_angles", [](py::buffer theta, py::buffer phi,
                             py::buffer pa, py::buffer out, bool IAU) {
            pybuffer_check_1D <double> (theta);
            pybuffer_check_1D <double> (phi);
            pybuffer_check_1D <double> (pa);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_theta = theta.request();
            py::buffer_info info_phi = phi.request();
            py::buffer_info info_pa = pa.request();
            py::buffer_info info_out = out.request();
            size_t ntheta = info_theta.size;
            size_t nphi = info_phi.size;
            size_t npa = info_pa.size;
            size_t nout = (size_t)(info_out.size / 4);
            if ((ntheta != nphi) || (ntheta != npa) || (ntheta != nout)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * thetaraw = reinterpret_cast <double *> (info_theta.ptr);
            double * phiraw = reinterpret_cast <double *> (info_phi.ptr);
            double * paraw = reinterpret_cast <double *> (info_pa.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::qa_from_angles(ntheta, thetaraw, phiraw, paraw, outraw,
                                  IAU);
            return;
        }, py::arg("theta"), py::arg("phi"), py::arg("pa"), py::arg("out"),
        py::arg(
            "IAU") = false, R"(
        Create quaternions from spherical coordinates and position angle.

        The theta angle is measured down from the North pole and phi is
        measured from the prime meridian.  The position angle is with respect
        to the local meridian at the point described by the theta / phi
        coordinates.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            theta (array_like):  flattened 1D array of float64 values.
            phi (array_like):  flattened 1D array of float64 values.
            pa (array_like):  flattened 1D array of float64 values.
            out (array_like):  flattened 1D array of float64 values.
            IAU (bool):  if True, use IAU convention.

        Returns:
            None

    )");

    m.def(
        "qa_to_angles", [](py::buffer q_in, py::buffer theta, py::buffer phi,
                           py::buffer pa, bool IAU) {
            pybuffer_check_1D <double> (theta);
            pybuffer_check_1D <double> (phi);
            pybuffer_check_1D <double> (pa);
            pybuffer_check_1D <double> (q_in);
            py::buffer_info info_theta = theta.request();
            py::buffer_info info_phi = phi.request();
            py::buffer_info info_pa = pa.request();
            py::buffer_info info_qin = q_in.request();
            size_t ntheta = info_theta.size;
            size_t nphi = info_phi.size;
            size_t npa = info_pa.size;
            size_t nq = (size_t)(info_qin.size / 4);
            if ((ntheta != nphi) || (ntheta != npa) || (ntheta != nq)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * thetaraw = reinterpret_cast <double *> (info_theta.ptr);
            double * phiraw = reinterpret_cast <double *> (info_phi.ptr);
            double * paraw = reinterpret_cast <double *> (info_pa.ptr);
            double * qinraw = reinterpret_cast <double *> (info_qin.ptr);
            toast::qa_to_angles(ntheta, qinraw, thetaraw, phiraw, paraw,
                                IAU);
            return;
        }, py::arg("q_in"), py::arg("theta"), py::arg("phi"), py::arg("pa"),
        py::arg(
            "IAU") = false, R"(
        Convert quaternions to spherical coordinates and position angle.

        The theta angle is measured down from the North pole and phi is
        measured from the prime meridian.  The position angle is with respect
        to the local meridian at the point described by the theta / phi
        coordinates.

        The results are stored in the output buffers.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            q_in (array_like):  flattened 1D array of float64 values.
            theta (array_like):  flattened 1D array of float64 values.
            phi (array_like):  flattened 1D array of float64 values.
            pa (array_like):  flattened 1D array of float64 values.
            IAU (bool):  if True, use IAU convention.

        Returns:
            None

    )");

    m.def(
        "qa_from_position", [](py::buffer theta, py::buffer phi,
                               py::buffer out) {
            pybuffer_check_1D <double> (theta);
            pybuffer_check_1D <double> (phi);
            pybuffer_check_1D <double> (out);
            py::buffer_info info_theta = theta.request();
            py::buffer_info info_phi = phi.request();
            py::buffer_info info_out = out.request();
            size_t ntheta = info_theta.size;
            size_t nphi = info_phi.size;
            size_t nout = (size_t)(info_out.size / 4);
            if ((ntheta != nphi) || (ntheta != nout)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * thetaraw = reinterpret_cast <double *> (info_theta.ptr);
            double * phiraw = reinterpret_cast <double *> (info_phi.ptr);
            double * outraw = reinterpret_cast <double *> (info_out.ptr);
            toast::qa_from_position(ntheta, thetaraw, phiraw, outraw);
            return;
        }, py::arg("theta"), py::arg("phi"), py::arg(
            "out"), R"(
        Create quaternions from spherical coordinates.

        The theta angle is measured down from the North pole and phi is
        measured from the prime meridian.

        The results are stored in the output buffer.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            theta (array_like):  flattened 1D array of float64 values.
            phi (array_like):  flattened 1D array of float64 values.
            out (array_like):  flattened 1D array of float64 values.

        Returns:
            None

    )");

    m.def(
        "qa_to_position", [](py::buffer q_in, py::buffer theta,
                             py::buffer phi) {
            pybuffer_check_1D <double> (theta);
            pybuffer_check_1D <double> (phi);
            pybuffer_check_1D <double> (q_in);
            py::buffer_info info_theta = theta.request();
            py::buffer_info info_phi = phi.request();
            py::buffer_info info_qin = q_in.request();
            size_t ntheta = info_theta.size;
            size_t nphi = info_phi.size;
            size_t nq = (size_t)(info_qin.size / 4);
            if ((ntheta != nphi) || (ntheta != nq)) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Buffer sizes are not consistent.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            double * thetaraw = reinterpret_cast <double *> (info_theta.ptr);
            double * phiraw = reinterpret_cast <double *> (info_phi.ptr);
            double * qinraw = reinterpret_cast <double *> (info_qin.ptr);
            toast::qa_to_position(ntheta, qinraw, thetaraw, phiraw);
            return;
        }, py::arg("q_in"), py::arg("theta"), py::arg(
            "phi"), R"(
        Convert quaternions to spherical coordinates.

        The theta angle is measured down from the North pole and phi is
        measured from the prime meridian.

        The results are stored in the output buffers.  To guarantee SIMD
        vectorization, the input and output arrays should be aligned
        (i.e. use an AlignedF64).

        Args:
            q_in (array_like):  flattened 1D array of float64 values.
            theta (array_like):  flattened 1D array of float64 values.
            phi (array_like):  flattened 1D array of float64 values.
            IAU (bool):  if True, use IAU convention.

        Returns:
            None

    )");

    return;
}
