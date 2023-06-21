// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef LIBTOAST_ACCELERATOR_HPP
#define LIBTOAST_ACCELERATOR_HPP

#include <common.hpp>

#ifdef _OPENMP
# include <omp.h>
#endif // ifdef _OPENMP


// Declarations for OpenMP target offload helpers.

typedef struct {
    // Pointers to the beginning and end of the block
    void * start;
    void * end;

    // Size of the usable block in bytes, not including alignment padding.
    size_t size;

    // Size including alignment
    size_t aligned_size;

    // Whether that block has been freed
    bool is_free;
} OmpBlock;


class OmpPoolResource {
    public:

        OmpPoolResource(int target, size_t size, size_t align = 256);
        OmpPoolResource() : OmpPoolResource(-1, 0) {}

        OmpPoolResource(int target) : OmpPoolResource(target, 0) {}

        OmpPoolResource(OmpPoolResource const &) = delete;

        ~OmpPoolResource();

        void release();

        void * allocate(std::size_t bytes);
        void deallocate(void * p);

    private:

        int target_;
        size_t pool_size_;
        void * raw_;
        size_t pool_used_;
        size_t alignment_;

        // Block handling
        std::vector <OmpBlock> blocks_;
        std::unordered_map <void *, size_t> ptr_to_block_;

        void alloc();
        bool verbose();
        size_t block_index(void * ptr);
        size_t compute_aligned_bytes(size_t bytes, size_t alignment);
        void * shift_void_pointer(void * ptr, size_t offset_bytes);

        // Remove free blocks
        void garbage_collection();
};


class OmpManager {
    public:

        static OmpManager & get();

        void assign_device(int node_procs, int node_rank, float mem_gb,
                           bool disabled = false);
        int get_device();
        bool device_is_host();

        void * create(void * buffer, size_t nbytes,
                      std::string const & name = std::string("NA"));
        void remove(void * buffer, size_t nbytes,
                    std::string const & name = std::string("NA"));
        void update_device(void * buffer, size_t nbytes,
                           std::string const & name = std::string("NA"));
        void update_host(void * buffer, size_t nbytes,
                         std::string const & name = std::string("NA"));

        void reset(void * buffer, size_t nbytes,
                   std::string const & name = std::string("NA"));

        int present(void * buffer, size_t nbytes,
                    std::string const & name = std::string("NA"));

        void dump();

        ~OmpManager();

        template <typename T>
        T * null_ptr() {
            static T instance = T();
            if (!present(static_cast <void *> (&instance), sizeof(T))) {
                // Create device copy on demand
                void * dummy = create(
                    static_cast <void *> (&instance), sizeof(T),
                    std::string("NULL")
                );
            }
            return &instance;
        }

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

        std::unordered_map <void *, size_t> mem_size_;
        std::unordered_map <void *, void *> mem_;
        std::unordered_map <void *, std::string> mem_name_;
        int host_dev_;
        int target_dev_;
        int node_procs_;
        int node_rank_;

        OmpPoolResource * pool_;
};


#endif // ifndef LIBTOAST_ACCELERATOR_HPP
