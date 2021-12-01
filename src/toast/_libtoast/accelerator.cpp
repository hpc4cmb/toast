// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#ifdef HAVE_OPENACC
# include <openacc.h>
#endif // ifdef HAVE_OPENACC


// FIXME: add registration function here to add stubs if openacc disabled

void init_accelerator(py::module & m) {

    // FIXME:  add ifdef around this.

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

    m.def(
        "acc_is_present", [](py::buffer data) {
            py::buffer_info info = data.request();
            void * p_host = reinterpret_cast <void *> (info.ptr);
            size_t bytes = (size_t)(info.size);
            auto result = acc_is_present(p_host, bytes);

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
            void * p_host = reinterpret_cast <void *> (info.ptr);
            size_t bytes = (size_t)(info.size);
            auto p_device = acc_copyin(p_host, bytes);

            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "copyin host pointer " << p_host << " (" << bytes << " bytes) on device at " << p_device;
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
            void * p_host = reinterpret_cast <void *> (info.ptr);
            size_t bytes = (size_t)(info.size);
            bool present = acc_is_present(p_host, bytes);
            if (! present) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Data is not present on device, cannot copy out.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "copyout host pointer " << p_host << " (" << bytes << " bytes) from device";
            log.verbose(o.str().c_str());

            acc_copyout(p_host, bytes);
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
            void * p_host = reinterpret_cast <void *> (info.ptr);
            size_t bytes = (size_t)(info.size);
            bool present = acc_is_present(p_host, bytes);
            if (! present) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Data is not present on device, cannot update.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "update device with host pointer " << p_host << " (" << bytes << " bytes)";
            log.verbose(o.str().c_str());

            acc_update_device(p_host, bytes);
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
            void * p_host = reinterpret_cast <void *> (info.ptr);
            size_t bytes = (size_t)(info.size);
            bool present = acc_is_present(p_host, bytes);
            if (! present) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Data is not present on device, cannot update.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "update host/self with host pointer " << p_host << " (" << bytes << " bytes)";
            log.verbose(o.str().c_str());

            acc_update_self(p_host, bytes);
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
            void * p_host = reinterpret_cast <void *> (info.ptr);
            size_t bytes = (size_t)(info.size);
            bool present = acc_is_present(p_host, bytes);
            if (! present) {
                auto log = toast::Logger::get();
                std::ostringstream o;
                o << "Data is not present on device, cannot delete.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "delete device mem for host pointer " << p_host << " (" << bytes << " bytes)";
            log.verbose(o.str().c_str());

            acc_delete(p_host, bytes);
            return;
        }, py::arg(
            "data"), R"(
        Delete the device copy of the data.

        Args:
            data (array):  The host data.

        Returns:
            None

    )");

    // Small test code used by the unit tests.

    m.def(
        "test_acc_op", [](py::buffer data, size_t n_det) {
            pybuffer_check_1D <double> (data);
            py::buffer_info info = data.request();
            double * raw = reinterpret_cast <double *> (info.ptr);
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

}
