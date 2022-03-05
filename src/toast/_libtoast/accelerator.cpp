// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>
#include <accelerator.hpp>

#include <cstring>
#include <cstdlib>
#include <utility>


OmpManager & OmpManager::get() {
    static OmpManager instance;
    return instance;
}

void OmpManager::assign_device(int node_procs, int node_rank) {
    std::ostringstream o;
    auto log = toast::Logger::get();
    if ((node_procs < 1) || (node_rank < 0)) {
        o << "OmpManager:  must have at least one process per node with a rank >= 0";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    if (node_rank >= node_procs) {
        o.str("");
        o << "OmpManager:  node rank must be < number of node procs";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    // Clear any existing memory buffers
    clear();

    node_procs_ = node_procs;
    node_rank_ = node_rank;

    // Number of targets
    int n_target = 0;
    #ifdef HAVE_OPENMP_TARGET
    n_target = omp_get_num_devices();
    #endif

    // If we have no accelerator (target) devices, assign all processes to
    // the host device.
    if (n_target == 0) {
        target_dev_ = host_dev_;
        o.str("");
        o << "OmpManager:  rank " << node_rank << " with " << node_procs
        << " processes per node, no target devices available, assigning to host device";
        log.verbose(o.str().c_str());
    } else {
        int proc_per_dev = (int)(node_procs / n_target);
        if (n_target * proc_per_dev < node_procs) {
            proc_per_dev += 1;
        }
        target_dev_ = (int)(node_rank / proc_per_dev);
        o.str("");
        o << "OmpManager:  rank " << node_rank << " with " << node_procs
        << " processes per node, using device " << target_dev_ << "("
        << n_target << " total)";
        log.verbose(o.str().c_str());
    }

    return;
}

int OmpManager::get_device() {
    std::ostringstream o;
    auto log = toast::Logger::get();
    if (target_dev_ < 0) {
        o << "OmpManager:  device not yet assigned, call assign_device() first";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    return target_dev_;
}

bool OmpManager::device_is_host() {
    if (target_dev_ == host_dev_) {
        return true;
    } else {
        return false;
    }
}

void * OmpManager::create(void * buffer, size_t nbytes) {
    size_t n = mem_size_.count(buffer);
    std::ostringstream o;
    auto log = toast::Logger::get();

    #ifdef HAVE_OPENMP_TARGET

    if (n != 0) {
        o << "OmpManager:  on create, host ptr " << buffer
        << " with " << nbytes << " bytes is already present "
        << "with " << mem_size_.at(buffer) << " bytes on device "
        << target_dev_;
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    // Add to the map
    o.str("");
    o << "OmpManager:  creating entry for host ptr "
    << buffer << " with " << nbytes << " bytes on device "
    << target_dev_;
    log.verbose(o.str().c_str());
    mem_size_[buffer] = nbytes;
    mem_[buffer] = omp_target_alloc(nbytes, target_dev_);
    if (mem_.at(buffer) == NULL) {
        o.str("");
        o << "OmpManager:  on create, host ptr " << buffer
        << " with " << nbytes << " bytes on device "
        << target_dev_ << ", allocation failed";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    int failed = omp_target_associate_ptr(buffer, mem_.at(buffer), mem_size_.at(buffer), 0, target_dev_);
    if (failed != 0) {
        o.str("");
        o << "OmpManager:  on create, host ptr " << buffer
        << " with " << nbytes << " bytes on device "
        << target_dev_ << ", failed to associate device ptr";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    return mem_.at(buffer);

    #else

    o << "OmpManager:  OpenMP target support disabled";
    log.error(o.str().c_str());
    throw std::runtime_error(o.str().c_str());

    #endif
}

void OmpManager::remove(void * buffer, size_t nbytes) {
    size_t n = mem_size_.count(buffer);
    auto log = toast::Logger::get();
    std::ostringstream o;

    #ifdef HAVE_OPENMP_TARGET

    if (n == 0) {
        o.str("");
        o << "OmpManager:  host ptr " << buffer
        << " is not present- cannot delete";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        size_t nb = mem_size_.at(buffer);
        if (nb != nbytes) {
            o.str("");
            o << "OmpManager:  on delete, host ptr " << buffer << " has "
            << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    o.str("");
    o << "OmpManager:  removing entry for host ptr "
    << buffer << " with " << nbytes << " bytes on device "
    << target_dev_;
    log.verbose(o.str().c_str());
    // First disassociate pointer
    int failed = omp_target_disassociate_ptr(buffer, target_dev_);
    if (failed != 0) {
        o.str("");
        o << "OmpManager:  on removal of host ptr " << buffer
        << " with " << nbytes << " bytes on device "
        << target_dev_ << ", failed to disassociate device ptr";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    mem_size_.erase(buffer);
    omp_target_free(mem_.at(buffer), target_dev_);
    mem_.erase(buffer);

    #else

    o << "OmpManager:  OpenMP target support disabled";
    log.error(o.str().c_str());
    throw std::runtime_error(o.str().c_str());

    #endif
}

void OmpManager::update_device(void * buffer, size_t nbytes) {
    size_t n = mem_size_.count(buffer);
    auto log = toast::Logger::get();
    std::ostringstream o;

    #ifdef HAVE_OPENMP_TARGET

    if (n == 0) {
        o.str("");
        o << "OmpManager:  host ptr " << buffer
        << " is not present- cannot update device";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        size_t nb = mem_size_.at(buffer);
        if (nb < nbytes) {
            o.str("");
            o << "OmpManager:  on update device, host ptr " << buffer << " has "
            << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    void * dev_buffer = mem_.at(buffer);
    o.str("");
    o << "OmpManager:  update device ptr " << dev_buffer << " from host ptr "
    << buffer << " with " << nbytes;
    int failed = omp_target_memcpy(dev_buffer, buffer, nbytes, 0, 0, target_dev_, host_dev_);
    if (failed != 0) {
        o.str("");
        o << "OmpManager:  copy of host ptr " << buffer
        << " with " << nbytes << " bytes to device "
        << target_dev_ << ", failed target memcpy";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }

    #else

    o << "OmpManager:  OpenMP target support disabled";
    log.error(o.str().c_str());
    throw std::runtime_error(o.str().c_str());

    #endif
}

void OmpManager::update_host(void * buffer, size_t nbytes) {
    size_t n = mem_size_.count(buffer);
    auto log = toast::Logger::get();
    std::ostringstream o;

    #ifdef HAVE_OPENMP_TARGET

    if (n == 0) {
        o.str("");
        o << "OmpManager:  host ptr " << buffer
        << " is not present- cannot update host";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        size_t nb = mem_size_.at(buffer);
        if (nb < nbytes) {
            o.str("");
            o << "OmpManager:  on update host, host ptr " << buffer << " has "
            << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    void * dev_buffer = mem_.at(buffer);
    o.str("");
    o << "OmpManager:  update host ptr " << buffer << " from device ptr "
    << dev_buffer << " with " << nbytes;
    int failed = omp_target_memcpy(buffer, dev_buffer, nbytes, 0, 0, host_dev_, target_dev_);
    if (failed != 0) {
        o.str("");
        o << "OmpManager:  copy of dev ptr " << dev_buffer
        << " with " << nbytes << " bytes from device "
        << target_dev_ << ", failed target memcpy";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }

    #else

    o << "OmpManager:  OpenMP target support disabled";
    log.error(o.str().c_str());
    throw std::runtime_error(o.str().c_str());

    #endif
}

int OmpManager::present(void * buffer, size_t nbytes) {
    #ifdef HAVE_OPENMP_TARGET

    size_t n = mem_.count(buffer);
    if (n == 0) {
        return 0;
    } else {
        size_t nb = mem_size_.at(buffer);
        if (nb != nbytes) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "OmpManager:  host ptr " << buffer << " is present"
            << ", but has " << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
        return 1;
    }

    #else

    o << "OmpManager:  OpenMP target support disabled";
    log.error(o.str().c_str());
    throw std::runtime_error(o.str().c_str());

    #endif
}

void * OmpManager::device_ptr(void * buffer) {
    auto log = toast::Logger::get();
    std::ostringstream o;

    #ifdef HAVE_OPENMP_TARGET

    size_t n = mem_.count(buffer);
    if (n == 0) {
        o.str("");
        o << "OmpManager:  host ptr " << buffer
        << " is not present- cannot get device pointer";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    return mem_.at(buffer);

    #else

    o << "OmpManager:  OpenMP target support disabled";
    log.error(o.str().c_str());
    throw std::runtime_error(o.str().c_str());

    #endif
}

void OmpManager::dump() {
    #ifdef HAVE_OPENMP_TARGET
    for(auto & p : mem_size_) {
        void * dev = mem_.at(p.first);
        std::cout << "OmpManager table:  " << p.first << ": "
        << p.second << " bytes on device " << target_dev_
        << " at " << dev << std::endl;
    }
    #endif
    return;
}

OmpManager::OmpManager() : mem_size_(), mem_() {
    host_dev_ = -1;
    #ifdef HAVE_OPENMP_TARGET
    host_dev_ = omp_get_initial_device();
    #endif
    target_dev_ = -1;
    node_procs_ = -1;
    node_rank_ = -1;
}

OmpManager::~OmpManager() {
    clear();
}

void OmpManager::clear() {
    #ifdef HAVE_OPENMP_TARGET
    for(auto & p : mem_) {
        omp_target_free(p.second, target_dev_);
    }
    #endif
    mem_size_.clear();
    mem_.clear();
}


void extract_accel_buffer(py::buffer_info const & info, void ** host_ptr,
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
        "accel_assign_device", [](int node_procs, int node_rank)
        {
            auto & log = toast::Logger::get();

            std::ostringstream o;
            #ifndef HAVE_OPENMP_TARGET
            o.str("");
            o << "TOAST not built with OpenMP target support";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
            #endif

            auto & omgr = OmpManager::get();
            omgr.assign_device(node_procs, node_rank);

            return;
        },
        R"(
            Assign processes to OpenMP target devices.
        )");

    m.def(
        "accel_get_device", []()
        {
            auto & log = toast::Logger::get();

            std::ostringstream o;
            #ifndef HAVE_OPENMP_TARGET
            o.str("");
            o << "TOAST not built with OpenMP target support";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
            #endif

            auto & omgr = OmpManager::get();
            return omgr.get_device();
        },
        R"(
            Return the OpenMP target device assigned to this process.
        )");

    m.def(
        "accel_present", [](py::buffer data)
        {
            auto & log = toast::Logger::get();
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_accel_buffer(info, &p_host, &n_elem, &n_bytes);

            std::ostringstream o;
            #ifndef HAVE_OPENMP_TARGET
            o.str("");
            o << "TOAST not built with OpenMP target support";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
            #endif

            auto & omgr = OmpManager::get();
            int result = omgr.present(p_host, n_bytes);

            o.str("");
            o << "host pointer " << p_host << " is_present = " << result;
            log.verbose(o.str().c_str());

            if (result == 0) {
                return false;
            } else {
                return true;
            }
        },
        py::arg("data"),
        R"(
        Check if the specified array is present on the accelerator device.

        Args:
            data (array):  The data array

        Returns:
            (bool):  True if the data is present, else False.

    )");

    m.def(
        "accel_create", [](py::buffer data)
        {
            auto & log = toast::Logger::get();
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_accel_buffer(info, &p_host, &n_elem, &n_bytes);

            std::ostringstream o;
            #ifndef HAVE_OPENMP_TARGET
            o.str("");
            o << "TOAST not built with OpenMP target support";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
            #endif

            auto & omgr = OmpManager::get();
            int present = omgr.present(p_host, n_bytes);
            if (present == 1) {
                o.str("");
                o << "Data is already present on device, cannot create.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            o.str("");
            o << "create device data with host pointer " << p_host << " (" << n_bytes << " bytes)";
            log.verbose(o.str().c_str());

            void * dev_mem = omgr.create(p_host, n_bytes);
            return;
        },
        py::arg("data"),
        R"(
        Create device copy of the data.

        Args:
            data (array):  The host data.

        Returns:
            None

    )");

    m.def(
        "accel_update_device", [](py::buffer data)
        {
            auto & log = toast::Logger::get();
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_accel_buffer(info, &p_host, &n_elem, &n_bytes);

            std::ostringstream o;
            #ifndef HAVE_OPENMP_TARGET
            o.str("");
            o << "TOAST not built with OpenMP target support";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
            #endif

            auto & omgr = OmpManager::get();
            int present = omgr.present(p_host, n_bytes);
            if (present == 0) {
                o.str("");
                o << "Data is not present on device, cannot update.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            o.str("");
            o << "update device with host pointer " << p_host << " (" << n_bytes << " bytes)";
            log.verbose(o.str().c_str());

            omgr.update_device(p_host, n_bytes);
            return;
        },
        py::arg("data"),
        R"(
        Update device copy of the data from the host.

        Args:
            data (array):  The host data.

        Returns:
            None

    )");

    m.def(
        "accel_update_host", [](py::buffer data)
        {
            auto & log = toast::Logger::get();
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_accel_buffer(info, &p_host, &n_elem, &n_bytes);

            std::ostringstream o;
            #ifndef HAVE_OPENMP_TARGET
            o.str("");
            o << "TOAST not built with OpenMP target support";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
            #endif

            auto & omgr = OmpManager::get();
            int present = omgr.present(p_host, n_bytes);
            if (present == 0) {
                o.str("");
                o << "Data is not present on device, cannot update host.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            o.str("");
            o << "update host pointer " << p_host << " ("
            << n_bytes << " bytes) from device";
            log.verbose(o.str().c_str());

            omgr.update_host(p_host, n_bytes);
            return;
        },
        py::arg("data"),
        R"(
        Update host copy of the data from the device.

        Args:
            data (array):  The host data.

        Returns:
            None

    )");

    m.def(
        "accel_delete", [](py::buffer data)
        {
            auto & log = toast::Logger::get();
            py::buffer_info info = data.request();
            void * p_host;
            size_t n_elem;
            size_t n_bytes;
            extract_accel_buffer(info, &p_host, &n_elem, &n_bytes);

            std::ostringstream o;
            #ifndef HAVE_OPENMP_TARGET
            o.str("");
            o << "TOAST not built with OpenMP target support";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
            #endif

            auto & omgr = OmpManager::get();
            int present = omgr.present(p_host, n_bytes);
            if (present == 0) {
                o.str("");
                o << "Data is not present on device, cannot delete.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            o.str("");
            o << "Delete device memory for host pointer " << p_host << " (" << n_bytes << " bytes)";
            log.verbose(o.str().c_str());

            omgr.remove(p_host, n_bytes);
            return;
        },
        py::arg("data"),
        R"(
        Delete the device copy of the data.

        Args:
            data (array):  The host data.

        Returns:
            None

    )");

    // FIXME: Small accelerator test code used by the unit tests.  Once the rest of
    // the compiled unit tests are consolidated within the bindings, we could put these
    // in there.

    m.def(
        "test_accel_op_buffer", [](py::buffer data)
        {
            // This is used to return the actual shape of each buffer
            std::vector <size_t> temp_shape(2);

            double * raw = extract_buffer <double> (
                data, "data", 2, temp_shape, {-1, -1}
            );
            size_t n_det = temp_shape[0];
            size_t n_samp = temp_shape[1];

            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = ! omgr.device_is_host();
            void * dev_raw = omgr.device_ptr(raw);

            #pragma omp target data device(dev) use_device_ptr(dev_raw) if(offload)
            {
                #pragma omp target teams distribute parallel for collapse(2)
                for (size_t i = 0; i < n_det; i++) {
                    for (size_t j = 0; j < n_samp; j++) {
                        raw[i * n_samp + j] *= 2.0;
                    }
                }
            }
            return;
        });

    m.def(
        "test_accel_op_array", [](py::array_t <double, py::array::c_style> data)
        {
            auto fast_data = data.mutable_unchecked <2>();
            double * raw = fast_data.mutable_data(0, 0);

            size_t n_det = fast_data.shape(0);
            size_t n_samp = fast_data.shape(1);

            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = ! omgr.device_is_host();
            void * dev_raw = omgr.device_ptr(raw);

            #pragma omp target data device(dev) use_device_ptr(dev_raw) if(offload)
            {
                #pragma omp target teams distribute parallel for collapse(2)
                for (size_t i = 0; i < n_det; i++) {
                    for (size_t j = 0; j < n_samp; j++) {
                        raw[i * n_samp + j] *= 2.0;
                    }
                }
            }
            return;
        });
}
