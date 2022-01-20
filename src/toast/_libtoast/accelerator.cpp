// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <toast/gpu_helpers.hpp>

#include <cstring>
#include <cstdlib>
#include <utility>

#ifdef HAVE_OPENACC
# include <openacc.h>
#endif // ifdef HAVE_OPENACC

#ifdef HAVE_CUDALIBS
# include <cuda_runtime_api.h>
#endif // ifdef HAVE_CUDALIBS


void extract_buffer_info(py::buffer_info const & info, void ** host_ptr,
                         size_t * n_elem, size_t * n_bytes) {
    (*host_ptr) = info.ptr; // reinterpret_cast <void *> (info.ptr);
    (*n_elem) = 1;
    for (py::ssize_t d = 0; d < info.ndim; d++) {
        // std::cerr << "acc buffer info dim " << d << " shape " << info.shape[d] << "
        // stride " << info.strides[d] << " (itemsize = " << info.itemsize << ") raw = "
        // << (*host_ptr) << std::endl;
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


bool fake_openacc() {
    // See if we should use the OpenACC code path, even when disabled
    // (Useful for debugging).
    bool ret = false;
    char * envval = ::getenv("TOAST_FAKE_OPENACC");
    if (envval != NULL) {
        ret = true;
    }
    return ret;
}


void init_accelerator(py::module & m) {
    m.def(
        "acc_enabled", []()
        {
            #ifdef HAVE_OPENACC
            return true;
            #else // ifdef HAVE_OPENACC
            return fake_openacc();
            #endif // ifdef HAVE_OPENACC
        },
        R"(
            Return True if TOAST was compiled with OpenACC support.
        )");

    m.def(
        "acc_is_fake", []()
        {
            #ifdef HAVE_OPENACC
            return false;
            #else // ifdef HAVE_OPENACC
            return fake_openacc();
            #endif // ifdef HAVE_OPENACC
        },
        R"(
            Return True if TOAST is using fake OpenACC.
        )");

    m.def(
        "acc_get_num_devices", []()
        {
            auto & log = toast::Logger::get();
            std::ostringstream o;
            #ifdef HAVE_OPENACC
            int nacc = acc_get_num_devices(acc_device_not_host);
            o << "OpenACC has " << nacc << " accelerators";
            log.verbose(o.str().c_str());
            return nacc;

            #else // ifdef HAVE_OPENACC
            # ifdef HAVE_CUDALIBS
            int ncuda;
            auto ret = cudaGetDeviceCount(&ncuda);
            o << "CUDA has " << ncuda << " accelerators";
            log.verbose(o.str().c_str());
            return ncuda;

            # else // ifdef HAVE_CUDALIBS
            if (fake_openacc()) {
                o << "Using one fake OpenACC device";
                log.verbose(o.str().c_str());
                return 1;
            } else {
                o << "No OpenACC or CUDA devices found";
                log.verbose(o.str().c_str());
                return 0;
            }

            # endif // ifdef HAVE_CUDALIBS
            #endif  // ifdef HAVE_OPENACC
        },
        R"(
            Return the total number of OpenACC devices.
        )");

    m.def(
        "acc_is_present", [](py::buffer data)
        {
            auto & log = toast::Logger::get();
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_buffer_info(info, &p_host, &n_elem, &n_bytes);

            int result;

            #ifdef HAVE_OPENACC

            # ifdef USE_OPENACC_MEMPOOL
            auto & pool = GPU_memory_pool::get();
            bool test = pool.is_present(p_host);
            result = test ? 1 : 0;
            # else // ifdef USE_OPENACC_MEMPOOL
            result = acc_is_present(p_host, n_bytes);
            # endif// ifdef USE_OPENACC_MEMPOOL

            #else
            auto & fake = FakeMemPool::get();
            result = fake.present(p_host, n_bytes);
            #endif

            std::ostringstream o;
            o << "host pointer " << p_host << " is_present = " << result;
            log.verbose(o.str().c_str());

            if (result == 0) {
                return false;
            } else {
                return true;
            }
        },
        py::arg(
            "data"),
        R"(
        Check if the specified array is present on the accelerator device.

        Returns:
            (bool):  True if the data is present, else False.

    )");

    m.def(
        "acc_copyin", [](py::buffer data)
        {
            auto & log = toast::Logger::get();
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_buffer_info(info, &p_host, &n_elem, &n_bytes);

            #ifdef HAVE_OPENACC

            # ifdef USE_OPENACC_MEMPOOL
            auto & pool = GPU_memory_pool::get();
            auto p_device = pool.toDevice(static_cast <char *> (p_host), n_bytes);
            # else // ifdef USE_OPENACC_MEMPOOL
            auto p_device = acc_copyin(p_host,
                                       n_bytes);
            acc_update_device(p_host, n_bytes);
            # endif// ifdef USE_OPENACC_MEMPOOL

            #else
            auto & fake = FakeMemPool::get();
            auto p_device = fake.copyin(p_host, n_bytes);
            #endif

            std::ostringstream o;
            o << "copyin host pointer " << p_host << " (" << n_bytes << " bytes) on device at " << p_device;
            log.verbose(o.str().c_str());

            return;
        },
        py::arg(
            "data"),
        R"(
        Copy the input data to the device, creating it if necessary.

        Args:
            data (array):  The host data.

        Returns:
            None

    )");

    m.def(
        "acc_copyout", [](py::buffer data)
        {
            auto & log = toast::Logger::get();
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_buffer_info(info, &p_host, &n_elem, &n_bytes);

            #ifdef HAVE_OPENACC
            int present = acc_is_present(p_host, n_bytes);
            #else
            auto & fake = FakeMemPool::get();
            int present = fake.present(p_host, n_bytes);
            if (present == 0) {
                fake.dump();
            }
            #endif

            if (present == 0) {
                std::ostringstream o;
                o << "Data is not present on device, cannot copy out.";
                log.error(o.str().c_str());
                throw std::runtime_error(
                          o.str().c_str());
            }

            std::ostringstream o;
            o << "copyout host pointer " << p_host << " (" << n_bytes << " bytes) from device";
            log.verbose(o.str().c_str());

            #ifdef HAVE_OPENACC
            # ifdef USE_OPENACC_MEMPOOL
            auto & pool = GPU_memory_pool::get();
            pool.fromDevice(p_host);
            # else // ifdef USE_OPENACC_MEMPOOL
            acc_copyout(p_host, n_bytes);
            # endif// ifdef USE_OPENACC_MEMPOOL
            #else
            fake.copyout(p_host, n_bytes);
            #endif

            return;
        },
        py::arg(
            "data"),
        R"(
        Copy device data into the host array.

        Args:
            data (array):  The host data.

        Returns:
            None

    )");

    m.def(
        "acc_update_device", [](py::buffer data)
        {
            auto & log = toast::Logger::get();
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_buffer_info(info, &p_host, &n_elem, &n_bytes);

            #ifdef HAVE_OPENACC
            int present = acc_is_present(p_host, n_bytes);
            #else
            auto & fake = FakeMemPool::get();
            int present = fake.present(p_host, n_bytes);
            if (present == 0) {
                fake.dump();
            }
            #endif

            if (present == 0) {
                std::ostringstream o;
                o << "Data is not present on device, cannot update.";
                log.error(o.str().c_str());
                throw std::runtime_error(
                          o.str().c_str());
            }

            std::ostringstream o;
            o << "update device with host pointer " << p_host << " (" << n_bytes << " bytes)";
            log.verbose(o.str().c_str());

            #ifdef HAVE_OPENACC
            # ifdef USE_OPENACC_MEMPOOL
            auto & pool = GPU_memory_pool::get();
            pool.update_gpu_memory(p_host);
            # else // ifdef USE_OPENACC_MEMPOOL
            acc_update_device(p_host, n_bytes);
            # endif// ifdef USE_OPENACC_MEMPOOL
            #else
            fake.update_device(p_host, n_bytes);
            #endif
            return;
        },
        py::arg(
            "data"),
        R"(
        Update device copy of the data from the host.

        Args:
            data (array):  The host data.

        Returns:
            None

    )");

    m.def(
        "acc_update_self", [](py::buffer data)
        {
            auto & log = toast::Logger::get();
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_buffer_info(info, &p_host, &n_elem, &n_bytes);

            #ifdef HAVE_OPENACC
            int present = acc_is_present(p_host, n_bytes);
            #else
            auto & fake = FakeMemPool::get();
            int present = fake.present(p_host, n_bytes);
            if (present == 0) {
                fake.dump();
            }
            #endif

            if (present == 0) {
                std::ostringstream o;
                o << "Data is not present on device, cannot update host.";
                log.error(o.str().c_str());
                throw std::runtime_error(
                          o.str().c_str());
            }

            std::ostringstream o;
            o << "update host/self with host pointer " << p_host << " (" << n_bytes << " bytes)";
            log.verbose(o.str().c_str());

            #ifdef HAVE_OPENACC
            # ifdef USE_OPENACC_MEMPOOL
            auto & pool = GPU_memory_pool::get();
            pool.update_cpu_memory(p_host);
            # else // ifdef USE_OPENACC_MEMPOOL
            acc_update_self(p_host, n_bytes);
            # endif// ifdef USE_OPENACC_MEMPOOL
            #else
            fake.update_self(p_host, n_bytes);
            #endif
            return;
        },
        py::arg(
            "data"),
        R"(
        Update host copy of the data from the device.

        Args:
            data (array):  The host data.

        Returns:
            None

    )");

    m.def(
        "acc_delete", [](py::buffer data)
        {
            auto & log = toast::Logger::get();
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_buffer_info(info, &p_host, &n_elem, &n_bytes);

            #ifdef HAVE_OPENACC
            int present = acc_is_present(p_host, n_bytes);
            #else
            auto & fake = FakeMemPool::get();
            int present = fake.present(p_host, n_bytes);
            if (present == 0) {
                fake.dump();
            }
            #endif

            if (present == 0) {
                std::ostringstream o;
                o << "Data is not present on device, cannot delete.";
                log.error(o.str().c_str());
                throw std::runtime_error(
                          o.str().c_str());
            }

            std::ostringstream o;
            o << "delete device mem for host pointer " << p_host << " (" << n_bytes << " bytes)";
            log.verbose(o.str().c_str());

            #ifdef HAVE_OPENACC

            # ifdef USE_OPENACC_MEMPOOL
            auto & pool = GPU_memory_pool::get();
            pool.free_associated_memory(p_host);
            # else // ifdef USE_OPENACC_MEMPOOL
            acc_delete(p_host, n_bytes);
            # endif// ifdef USE_OPENACC_MEMPOOL

            #else
            fake.remove(p_host, n_bytes);
            #endif
            return;
        },
        py::arg(
            "data"),
        R"(
        Delete the device copy of the data.

        Args:
            data (array):  The host data.

        Returns:
            None

    )");

    // Small test code used by the unit tests.

    m.def(
        "test_acc_op_buffer", [](py::buffer data, size_t n_det)
        {
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
        "test_acc_op_array", [](py::array_t <double, py::array::c_style> data)
        {
            auto fast_data = data.mutable_unchecked <2>();
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
