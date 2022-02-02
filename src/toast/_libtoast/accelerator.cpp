// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>

#include <cstring>
#include <cstdlib>
#include <utility>

#ifdef _OPENMP
# include <omp.h>
#endif // ifdef _OPENMP



OmpMemMap & OmpMemMap::get() {
    static OmpMemMap instance;
    return instance;
}

void * OmpMemMap::create(void * buffer, size_t nbytes) {
    size_t n = mem_size.count(buffer);
    std::ostringstream o;
    auto log = toast::Logger::get();
    if (n != 0) {
        o << "OmpMemMap:  on create, host ptr " << buffer
        << " with " << nbytes << " bytes is already present "
        << "with " << mem_size.at(buffer) << " bytes";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    // Add to the map
    o.str("");
    o << "OmpMemMap:  creating entry for host ptr "
    << buffer << " with " << nbytes << " bytes";
    log.verbose(o.str().c_str());
    mem_size[buffer] = nbytes;
    mem[buffer] = malloc(nbytes);
    if (mem.at(buffer) == NULL) {
        o.str("");
        o << "OmpMemMap:  on create, host ptr " << buffer
        << " with " << nbytes << " bytes, device allocation failed";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    return mem.at(buffer);
}

void OmpMemMap::remove(void * buffer, size_t nbytes) {
    size_t n = mem_size.count(buffer);
    auto log = toast::Logger::get();
    std::ostringstream o;
    if (n == 0) {
        o.str("");
        o << "OmpMemMap:  host ptr " << buffer
        << " is not present- cannot delete";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        size_t nb = mem_size.at(buffer);
        if (nb != nbytes) {
            o.str("");
            o << "OmpMemMap:  on delete, host ptr " << buffer << " has "
            << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    o.str("");
    o << "OmpMemMap:  removing entry for host ptr "
    << buffer << " with " << nbytes << " bytes";
    log.verbose(o.str().c_str());
    mem_size.erase(buffer);
    free(mem.at(buffer));
    mem.erase(buffer);
}

void * OmpMemMap::copyin(void * buffer, size_t nbytes) {
    size_t n = mem_size.count(buffer);
    std::ostringstream o;
    auto log = toast::Logger::get();
    void * ptr;
    if (n == 0) {
        ptr = create(buffer, nbytes);
    } else {
        size_t nb = mem_size.at(buffer);
        if (nb < nbytes) {
            o.str("");
            o << "OmpMemMap:  on copyin, host ptr " << buffer << " has "
            << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    ptr = mem.at(buffer);
    o.str("");
    o << "OmpMemMap:  copy in host ptr "
    << buffer << " with " << nbytes << " bytes to device "
    << ptr;
    log.verbose(o.str().c_str());
    void * temp = memcpy(ptr, buffer, nbytes);
    return mem.at(buffer);
}

void OmpMemMap::copyout(void * buffer, size_t nbytes) {
    size_t n = mem_size.count(buffer);
    auto log = toast::Logger::get();
    std::ostringstream o;
    size_t nb;
    if (n == 0) {
        o.str("");
        o << "OmpMemMap:  host ptr " << buffer
        << " is not present- cannot copy out";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        nb = mem_size.at(buffer);
        if (nb < nbytes) {
            o.str("");
            o << "OmpMemMap:  on copyout, host ptr " << buffer << " has "
            << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    void * ptr = mem.at(buffer);
    o.str("");
    o << "OmpMemMap:  copy out host ptr "
    << buffer << " with " << nbytes << " bytes from device "
    << ptr;
    void * temp = memcpy(buffer, ptr, nbytes);
    // Even if we copyout a portion of the buffer, remove the full thing.
    remove(buffer, nb);
}

void OmpMemMap::update_device(void * buffer, size_t nbytes) {
    size_t n = mem_size.count(buffer);
    auto log = toast::Logger::get();
    std::ostringstream o;
    if (n == 0) {
        o.str("");
        o << "OmpMemMap:  host ptr " << buffer
        << " is not present- cannot update device";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        size_t nb = mem_size.at(buffer);
        if (nb < nbytes) {
            o.str("");
            o << "OmpMemMap:  on update device, host ptr " << buffer << " has "
            << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    void * dev = mem.at(buffer);
    o.str("");
    o << "OmpMemMap:  update device from host ptr "
    << buffer << " with " << nbytes << " to " << dev;
    void * temp = memcpy(dev, buffer, nbytes);
}

void OmpMemMap::update_self(void * buffer, size_t nbytes) {
    size_t n = mem_size.count(buffer);
    auto log = toast::Logger::get();
    std::ostringstream o;
    if (n == 0) {
        o.str("");
        o << "OmpMemMap:  host ptr " << buffer
        << " is not present- cannot update host";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        size_t nb = mem_size.at(buffer);
        if (nb < nbytes) {
            o.str("");
            o << "OmpMemMap:  on update self, host ptr " << buffer << " has "
            << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    void * dev = mem.at(buffer);
    o.str("");
    o << "OmpMemMap:  update host ptr "
    << buffer << " with " << nbytes << " from " << dev;
    void * temp = memcpy(buffer, dev, nbytes);
}

// Use int instead of bool to match the acc_is_present API
int OmpMemMap::present(void * buffer, size_t nbytes) {
    size_t n = mem_size.count(buffer);
    if (n == 0) {
        return 0;
    } else {
        size_t nb = mem_size.at(buffer);
        if (nb != nbytes) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "OmpMemMap:  host ptr " << buffer << " is present"
            << ", but has " << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
        return 1;
    }
}

void * OmpMemMap::device_ptr(void * buffer) {
    auto log = toast::Logger::get();
    std::ostringstream o;
    size_t n = mem.count(buffer);
    if (n == 0) {
        o.str("");
        o << "OmpMemMap:  host ptr " << buffer
        << " is not present- cannot get device pointer";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    return mem.at(buffer);
}

void OmpMemMap::dump() {
    for(auto & p : mem_size) {
        void * dev = mem.at(p.first);
        std::cout << "OmpMemMap table:  " << p.first << ": "
        << p.second << " bytes on dev at " << dev << std::endl;
    }
    return;
}

OmpMemMap::OmpMemMap() : mem_size(), mem() {
}

OmpMemMap::~OmpMemMap() {
    for(auto & p : mem) {
        free(p.second);
    }
    mem_size.clear();
    mem.clear();
}


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


void init_accelerator(py::module & m) {
    m.def(
        "accel_enabled", []()
        {
            #ifdef HAVE_OPENMP_TARGET
            return true;
            #else
            return false;
            #endif
        },
        R"(
            Return True if TOAST was compiled with OpenMP target offload support.
        )");

    m.def(
        "accel_get_num_devices", []()
        {
            auto & log = toast::Logger::get();
            std::ostringstream o;
            int naccel = 0;
            #ifdef HAVE_OPENMP_TARGET
            naccel = omp_get_num_devices();
            #endif
            o << "OpenMP has " << naccel << " accelerator devices";
            log.verbose(o.str().c_str());
            return naccel;
        },
        R"(
            Return the total number of OpenMP target devices.
        )");

    m.def(
        "accel_is_present", [](py::buffer data, int device)
        {
            auto & log = toast::Logger::get();
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_buffer_info(info, &p_host, &n_elem, &n_bytes);

            int result = 0;

            #ifdef HAVE_OPENMP_TARGET
            result = omp_target_is_present(p_host, device);
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
        py::arg("data"), py::arg("device"),
        R"(
        Check if the specified array is present on the accelerator device.

        Args:
            data (array):  The data array
            device (int):  The accelerator device index

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
            auto & fake = OmpMemMap::get();
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
            auto & fake = OmpMemMap::get();
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
            auto & fake = OmpMemMap::get();
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
            auto & fake = OmpMemMap::get();
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
            auto & fake = OmpMemMap::get();
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
