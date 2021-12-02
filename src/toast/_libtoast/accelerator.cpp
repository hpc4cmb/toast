// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#ifdef HAVE_OPENACC
# include <openacc.h>
#endif // ifdef HAVE_OPENACC


// FIXME: add registration function here to add stubs if openacc disabled

void extract_buffer_info(py::buffer_info const & info, void ** host_ptr, size_t * n_elem, size_t * n_bytes) {
    (*host_ptr) = reinterpret_cast <void *> (info.ptr);
    (*n_elem) = 1;
    for (py::ssize_t d = 0; d < info.ndim; d++) {
        (*n_elem) *= info.shape[d];
        if (info.strides[d] != info.itemsize) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "Cannot use python buffers with stride != itemsize.";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    (*n_bytes) = (*n_elem) * info.itemsize;
    return;
}

void register_stub(py::module & m, char const * name) {
    m.def(name, [](py::buffer data) {
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "TOAST was not built with OpenACC support.";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    });
}


void init_accelerator(py::module & m) {

    m.def(
        "acc_enabled", []() {
            #ifdef HAVE_OPENACC
            return true;
            #else
            return false;
            #endif
        }, R"(
            Return True if TOAST was compiled with OpenACC support.
        )"
    );

    #ifdef HAVE_OPENACC

    m.def(
        "acc_is_present", [](py::buffer data) {
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_buffer_info(info, &p_host, &n_elem, &n_bytes);

            auto result = acc_is_present(p_host, n_bytes);

            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "host pointer " << p_host << " is_present = " << result;
            log.verbose(o.str().c_str());

            if (result == 0) {
                return false;
            } else {
                return true;
            }
        }, py::arg(
            "data"), R"(
        Check if the specified array is present on the accelerator device.

        Returns:
            (bool):  True if the data is present, else False.

    )");

    m.def(
        "acc_copyin", [](py::buffer data) {
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_buffer_info(info, &p_host, &n_elem, &n_bytes);

            auto p_device = acc_copyin(p_host, n_bytes);

            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "copyin host pointer " << p_host << " (" << n_bytes << " bytes) on device at " << p_device;
            log.verbose(o.str().c_str());

            return;
        }, py::arg(
            "data"), R"(
        Copy the input data to the device, creating it if necessary.

        Args:
            data (array):  The host data.

        Returns:
            None

    )");

    m.def(
        "acc_copyout", [](py::buffer data) {
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_buffer_info(info, &p_host, &n_elem, &n_bytes);

            bool present = acc_is_present(p_host, n_bytes);
            if (! present) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Data is not present on device, cannot copy out.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "copyout host pointer " << p_host << " (" << n_bytes << " bytes) from device";
            log.verbose(o.str().c_str());

            acc_copyout(p_host, n_bytes);
            return;
        }, py::arg(
            "data"), R"(
        Copy device data into the host array.

        Args:
            data (array):  The host data.

        Returns:
            None

    )");

    m.def(
        "acc_update_device", [](py::buffer data) {
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_buffer_info(info, &p_host, &n_elem, &n_bytes);

            bool present = acc_is_present(p_host, n_bytes);
            if (! present) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Data is not present on device, cannot update.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "update device with host pointer " << p_host << " (" << n_bytes << " bytes)";
            log.verbose(o.str().c_str());

            acc_update_device(p_host, n_bytes);
            return;
        }, py::arg(
            "data"), R"(
        Update device copy of the data from the host.

        Args:
            data (array):  The host data.

        Returns:
            None

    )");

    m.def(
        "acc_update_self", [](py::buffer data) {
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_buffer_info(info, &p_host, &n_elem, &n_bytes);

            bool present = acc_is_present(p_host, n_bytes);
            if (! present) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Data is not present on device, cannot update host.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "update host/self with host pointer " << p_host << " (" << n_bytes << " bytes)";
            log.verbose(o.str().c_str());

            acc_update_self(p_host, n_bytes);
            return;
        }, py::arg(
            "data"), R"(
        Update host copy of the data from the device.

        Args:
            data (array):  The host data.

        Returns:
            None

    )");

    m.def(
        "acc_delete", [](py::buffer data) {
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_buffer_info(info, &p_host, &n_elem, &n_bytes);

            bool present = acc_is_present(p_host, n_bytes);
            if (! present) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Data is not present on device, cannot delete.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "delete device mem for host pointer " << p_host << " (" << n_bytes << " bytes)";
            log.verbose(o.str().c_str());

            acc_delete(p_host, n_bytes);
            return;
        }, py::arg(
            "data"), R"(
        Delete the device copy of the data.

        Args:
            data (array):  The host data.

        Returns:
            None

    )");

    #else

    register_stub(m, "acc_is_present");
    register_stub(m, "acc_copyin");
    register_stub(m, "acc_copyout");
    register_stub(m, "acc_update_device");
    register_stub(m, "acc_update_self");
    register_stub(m, "acc_delete");

    #endif // HAVE_OPENACC

    // Small test code used by the unit tests.

    m.def(
        "test_acc_op_buffer", [](py::buffer data, size_t n_det) {
            pybuffer_check_1D <double> (data);
            py::buffer_info info = data.request();

            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_buffer_info(info, &p_host, &n_elem, &n_bytes);

            double * raw = reinterpret_cast <double *> (info.ptr);

            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "test_acc_op p_host = " << p_host << " (" << n_bytes << " bytes)";
            o << " cast to double * = " << raw;
            log.verbose(o.str().c_str());

            size_t n_total = (size_t)(info.size / sizeof(double));
            size_t n_samp = (size_t)(n_total / n_det);

            #pragma acc data present(raw)
            {
                #pragma acc parallel loop
                for (size_t i = 0; i < n_det; i++) {
                    for (size_t j = 0; j < n_samp; j++) {
                        raw[i * n_samp + j] *= 2.0;
                    }
                }
            }
            return;
        });

    m.def(
        "test_acc_op_array", [](py::array_t <double, py::array::c_style> data) {
            auto fast_data = data.mutable_unchecked<2>();
            double * raw = fast_data.mutable_data(0, 0);

            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "test_acc_op_array p_host = " << raw;
            log.verbose(o.str().c_str());

            size_t n_det = fast_data.shape(0);
            size_t n_samp = fast_data.shape(1);

            #pragma acc data present(raw)
            {
                #pragma acc parallel loop
                for (size_t i = 0; i < n_det; i++) {
                    for (size_t j = 0; j < n_samp; j++) {
                        raw[i * n_samp + j] *= 2.0;
                    }
                }
            }
            return;
        });

}
