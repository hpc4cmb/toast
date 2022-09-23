// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef LIBTOAST_ACCELERATOR_HPP
#define LIBTOAST_ACCELERATOR_HPP

#include <common.hpp>

#ifdef _OPENMP
# include <omp.h>
#endif // ifdef _OPENMP


// Declarations for OpenMP target offload helpers

// This helper class stores a mapping between host and device pointers.
// Once the OpenMP 5.1 standard is widely implemented across compilers,
// We can use the omp_get_mapped_ptr() method.


class OmpManager {
    public:

        static OmpManager & get();

        void assign_device(int node_procs, int node_rank, bool disabled = false);
        int get_device();
        bool device_is_host();

        void * create(void * buffer, size_t nbytes);
        void remove(void * buffer, size_t nbytes);
        void update_device(void * buffer, size_t nbytes);
        void update_host(void * buffer, size_t nbytes);

        int present(void * buffer, size_t nbytes);

        void dump();

        ~OmpManager();

        void * null;

        template <typename T>
        T * device_ptr(T * buffer) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            // If the device is the host device, return
            if (device_is_host()) {
                return buffer;
            }
            #ifdef HAVE_OPENMP_TARGET
            void * vbuffer = static_cast <void *> (buffer);
            size_t n = mem_.count(vbuffer);
            if (n == 0) {
                o.str("");
                o << "OmpManager:  host ptr " << buffer
                << " is not present- cannot get device pointer";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }
            return static_cast <T *> (mem_.at(vbuffer));
            #else // ifdef HAVE_OPENMP_TARGET
            o << "OmpManager:  OpenMP target support disabled";
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
            return NULL;
            #endif // ifdef HAVE_OPENMP_TARGET
        }

    private:

        OmpManager();
        void clear();
        void allocate_dummy(int n_target);
        void free_dummy();

        std::unordered_map <void *, size_t> mem_size_;
        std::unordered_map <void *, void *> mem_;
        int host_dev_;
        int target_dev_;
        int node_procs_;
        int node_rank_;
        void * dev_null_;
};


#endif // ifndef LIBTOAST_ACCELERATOR_HPP
