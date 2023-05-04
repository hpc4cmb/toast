// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <module.hpp>
#include <accelerator.hpp>

#include <cstring>
#include <cstdlib>
#include <utility>


OmpPoolResource::OmpPoolResource(int target, size_t size, size_t align) {
    if (target < 0) {
        #ifdef HAVE_OPENMP_TARGET
        target_ = omp_get_num_devices() - 1;
        #endif // ifdef HAVE_OPENMP_TARGET
    } else {
        target_ = target;
    }
    pool_size_ = size;
    alignment_ = align;
    alloc();
}

OmpPoolResource::~OmpPoolResource() {
    release();
}

bool OmpPoolResource::verbose() {
    // Helper function to check the log level once and return a static value.
    // This is useful to avoid expensive string operations inside deeply nested
    // functions.
    static bool called = false;
    static bool verbose = false;
    if (!called) {
        // First time we were called
        auto & env = toast::Environment::get();
        std::string logval = env.log_level();
        if (strncmp(logval.c_str(), "VERBOSE", 7) == 0) {
            verbose = true;
        }
        called = true;
    }
    return verbose;
}

void OmpPoolResource::release() {
    bool extra = verbose();

    // Reset our block containers and free device memory.
    blocks_.clear();
    ptr_to_block_.clear();
    if (raw_ != nullptr) {
        if (extra) {
            std::ostringstream o;
            auto log = toast::Logger::get();
            o << "  OmpPoolResource:  Free " << pool_size_ << " bytes on device " <<
                target_;
            o << " at " << raw_;
            log.verbose(o.str().c_str());
        }
        #ifdef HAVE_OPENMP_TARGET
        omp_target_free(raw_, target_);
        #endif // ifdef HAVE_OPENMP_TARGET
    }
    return;
}

void * OmpPoolResource::allocate(size_t bytes) {
    bool extra = verbose();

    // Compute the number of aligned bytes needed
    size_t aligned_bytes = compute_aligned_bytes(bytes, alignment_);

    // Check that we have available space
    if (pool_used_ + aligned_bytes > pool_size_) {
        std::ostringstream o;
        auto log = toast::Logger::get();
        o << "  OmpPoolResource:  request of " << aligned_bytes << " aligned bytes ";
        o << "would exceed pool size of " << pool_size_ << " bytes";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }

    // Build a block starting at the end of the existing blocks
    void * next_start = blocks_.back().end;
    OmpBlock next_block;
    next_block.start = next_start;
    next_block.end = shift_void_pointer(next_start, aligned_bytes);
    next_block.size = bytes;
    next_block.aligned_size = aligned_bytes;
    next_block.is_free = false;
    if (extra) {
        std::ostringstream o;
        auto log = toast::Logger::get();
        o << "  OmpPoolResource:  Append block " << next_block.start << ":";
        o << next_block.end << " with " << next_block.size << " bytes (";
        o << aligned_bytes << " aligned)";
        log.verbose(o.str().c_str());
    }

    // Store the block and its index
    size_t bindex = blocks_.size();
    ptr_to_block_[next_start] = bindex;
    blocks_.push_back(next_block);

    // Increment the used count
    pool_used_ += aligned_bytes;

    return next_start;
}

void OmpPoolResource::deallocate(void * p) {
    bool extra = verbose();

    // Get the memory block associated with this pointer
    size_t bindex = block_index(p);
    OmpBlock & block = blocks_[bindex];
    if (extra) {
        std::ostringstream o;
        auto log = toast::Logger::get();
        o << "  OmpPoolResource:  Delete block " << block.start << ":";
        o << block.end << " with " << block.size << " bytes (";
        o << block.aligned_size << " aligned)";
        log.verbose(o.str().c_str());
    }

    // Mark block as unused
    block.is_free = true;

    // Free trailing unused blocks
    garbage_collection();
}

void OmpPoolResource::alloc() {
    bool extra = verbose();

    if (pool_size_ == 0) {
        // Attempt to guess at the pool size
        pool_size_ = 1024 * 1024 * 1024;
        std::ostringstream o;
        auto log = toast::Logger::get();
        o << "  OmpPoolResource:  Using " << pool_size_;
        o << " bytes for default pool size";
        log.warning(o.str().c_str());
    }
    #ifdef HAVE_OPENMP_TARGET
    raw_ = omp_target_alloc(pool_size_, target_);
    #endif // ifdef HAVE_OPENMP_TARGET
    if (extra) {
        std::ostringstream o;
        auto log = toast::Logger::get();
        o << "  OmpPoolResource:  Allocated pool of " << pool_size_ << " bytes";
        o << " on target " << target_;
        log.verbose(o.str().c_str());
    }

    // Current bytes used
    pool_used_ = 0;

    // Large enough for any device?
    alignment_ = 512;

    // Initialize an empty starting block
    OmpBlock first_block;
    first_block.start = raw_;
    first_block.end = raw_;
    first_block.size = 0;
    first_block.aligned_size = 0;
    first_block.is_free = false;
    blocks_.push_back(first_block);
}

size_t OmpPoolResource::block_index(void * ptr) {
    auto entry = ptr_to_block_.find(ptr);
    if (entry == ptr_to_block_.end()) {
        std::ostringstream o;
        auto log = toast::Logger::get();
        o << "  OmpPoolResource:  Pointer " << ptr << " is not mapped to a block";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    return entry->second;
}

// Remove free blocks
void OmpPoolResource::garbage_collection() {
    bool extra = verbose();
    OmpBlock & block = blocks_.back();
    while (block.is_free) {
        // Remove pointer from map
        if (extra) {
            std::ostringstream o;
            auto log = toast::Logger::get();
            o << "  OmpPoolResource:  GC block " << block.start << ":";
            o << block.end << " with " << block.size << " bytes (";
            o << block.aligned_size << " aligned)";
            log.verbose(o.str().c_str());
        }
        ptr_to_block_.erase(block.start);

        // Shrink the currently used size
        pool_used_ -= block.aligned_size;

        // Delete block
        blocks_.pop_back();

        // Get next block
        block = blocks_.back();
    }
}

size_t OmpPoolResource::compute_aligned_bytes(size_t bytes, size_t alignment) {
    size_t aligned_blocks = (size_t)(bytes / alignment);
    if (aligned_blocks * alignment != bytes) {
        aligned_blocks += 1;
    }
    return aligned_blocks * alignment;
}

void * OmpPoolResource::shift_void_pointer(void * ptr, size_t offset_bytes) {
    uint8_t * bptr = static_cast <uint8_t *> (ptr);
    void * new_ptr = static_cast <void *> (&(bptr[offset_bytes]));
    return new_ptr;
}

OmpManager & OmpManager::get() {
    static OmpManager instance;
    return instance;
}

void OmpManager::assign_device(int node_procs, int node_rank, float mem_gb,
                               bool disabled) {
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
    #endif // ifdef HAVE_OPENMP_TARGET
    if (disabled) {
        n_target = 0;
    }

    // If we have no accelerator (target) devices, assign all processes to
    // the host device.
    int proc_per_dev = 0;

    if (n_target == 0) {
        target_dev_ = host_dev_;
        o.str("");
        o << "  OmpManager:  rank " << node_rank << " with " << node_procs
          << " processes per node, no target devices available, "
          << "assigning to host device";
        log.verbose(o.str().c_str());
    } else {
        proc_per_dev = (int)(node_procs / n_target);
        if (n_target * proc_per_dev < node_procs) {
            proc_per_dev += 1;
        }
        target_dev_ = (int)(node_rank / proc_per_dev);
        o.str("");
        o << "  OmpManager:  rank " << node_rank << " with " << node_procs
          << " processes per node, using device " << target_dev_ << " ("
          << n_target << " total)";
        log.verbose(o.str().c_str());
        omp_set_default_device(target_dev_);
    }

    auto & env = toast::Environment::get();
    env.set_acc(n_target, proc_per_dev, target_dev_);

    // Create a memory pool on the target device

    // if (n_target > 0) {
    //     double one_gb = 1024.0 * 1024.0 * 1024.0;
    //     size_t proc_bytes = (size_t)(mem_gb * one_gb / (double)proc_per_dev);
    //     if (pool_ != nullptr) {
    //         delete pool_;
    //     }
    //     pool_ = new OmpPoolResource(target_dev_, proc_bytes);
    // }

    return;
}

int OmpManager::get_device() {
    std::ostringstream o;
    auto log = toast::Logger::get();
    if (target_dev_ < -1) {
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

void * OmpManager::create(void * buffer, size_t nbytes, std::string const & name) {
    size_t n = mem_size_.count(buffer);
    std::ostringstream o;
    auto log = toast::Logger::get();

    // If the device is the host device, return
    if (device_is_host()) {
        return buffer;
    }

    #ifdef HAVE_OPENMP_TARGET

    if (n != 0) {
        o << "OmpManager:  on create, host ptr " << buffer
          << " with " << nbytes << " bytes (name='" << mem_name_.at(buffer)
          << "') is already present "
          << "with " << mem_size_.at(buffer) << " bytes on device "
          << target_dev_;
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }

    // Add to the map
    o.str("");
    o << "  OmpManager:  creating entry for host ptr "
      << buffer << " with " << nbytes << " bytes (name='" << name
      << "') on device " << target_dev_;
    log.verbose(o.str().c_str());
    mem_size_[buffer] = nbytes;
    mem_name_[buffer] = name;
    mem_[buffer] = omp_target_alloc(nbytes, target_dev_);

    // mem_[buffer] = pool_->allocate(nbytes);
    if (mem_.at(buffer) == NULL) {
        o.str("");
        o << "OmpManager:  on create, host ptr " << buffer
          << " with " << nbytes << " bytes (name='" << name
          << "') on device "
          << target_dev_ << ", allocation failed";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    int failed = omp_target_associate_ptr(
        buffer, mem_.at(buffer), mem_size_.at(buffer), 0, target_dev_
    );
    if (failed != 0) {
        o.str("");
        o << "OmpManager:  on create, host ptr " << buffer
          << " with " << nbytes << " bytes (name='" << name
          << "') on device "
          << target_dev_ << ", failed to associate device ptr";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    return mem_.at(buffer);

    #else // ifdef HAVE_OPENMP_TARGET

    o << "OmpManager:  OpenMP target support disabled";
    log.error(o.str().c_str());
    throw std::runtime_error(o.str().c_str());

    #endif // ifdef HAVE_OPENMP_TARGET
}

void OmpManager::remove(void * buffer, size_t nbytes, std::string const & name) {
    size_t n = mem_size_.count(buffer);
    auto log = toast::Logger::get();
    std::ostringstream o;

    // If the device is the host device, return
    if (device_is_host()) {
        return;
    }

    #ifdef HAVE_OPENMP_TARGET

    if (n == 0) {
        o.str("");
        o << "OmpManager:  host ptr " << buffer
          << " (name='" << name << "') is not present- cannot delete";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        size_t nb = mem_size_.at(buffer);
        if (name != mem_name_.at(buffer)) {
            o.str("");
            o << "OmpManager:  on delete, host ptr " << buffer
              << " has name '" << mem_name_.at(buffer) << "' in the table, not '"
              << name << "'";
            log.warning(o.str().c_str());
        }
        if (nb != nbytes) {
            o.str("");
            o << "OmpManager:  on delete, host ptr " << buffer
              << " (name='" << mem_name_.at(buffer) << "') has "
              << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    o.str("");
    o << "  OmpManager:  removing entry for host ptr "
      << buffer << " (name='" << mem_name_.at(buffer) << "') with "
      << nbytes << " bytes on device " << target_dev_;
    log.verbose(o.str().c_str());

    // First disassociate pointer
    int failed = omp_target_disassociate_ptr(buffer, target_dev_);
    if (failed != 0) {
        o.str("");
        o << "OmpManager:  on removal of host ptr " << buffer
          << " (name='" << mem_name_.at(buffer)
          << "') with " << nbytes << " bytes on device "
          << target_dev_ << ", failed to disassociate device ptr";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    mem_size_.erase(buffer);
    mem_name_.erase(buffer);

    // pool_->deallocate(mem_.at(buffer));
    omp_target_free(mem_.at(buffer), target_dev_);
    mem_.erase(buffer);

    #else // ifdef HAVE_OPENMP_TARGET

    o << "OmpManager:  OpenMP target support disabled";
    log.error(o.str().c_str());
    throw std::runtime_error(o.str().c_str());

    #endif // ifdef HAVE_OPENMP_TARGET
}

void OmpManager::update_device(void * buffer, size_t nbytes, std::string const & name) {
    size_t n = mem_size_.count(buffer);
    auto log = toast::Logger::get();
    std::ostringstream o;

    // If the device is the host device, return
    if (device_is_host()) {
        return;
    }

    #ifdef HAVE_OPENMP_TARGET

    if (n == 0) {
        o.str("");
        o << "OmpManager:  host ptr " << buffer
          << " (name='" << name
          << "') is not present- cannot update device";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        size_t nb = mem_size_.at(buffer);
        if (name != mem_name_.at(buffer)) {
            o.str("");
            o << "OmpManager:  on update device, host ptr " << buffer
              << " has name '" << mem_name_.at(buffer) << "' in the table, not '"
              << name << "'";
            log.warning(o.str().c_str());
        }
        if (nb < nbytes) {
            o.str("");
            o << "OmpManager:  on update device, host ptr " << buffer
              << " (name='" << mem_name_.at(buffer) << "') has "
              << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    void * dev_buffer = mem_.at(buffer);
    o.str("");
    o << "  OmpManager:  update device ptr " << dev_buffer
      << " from host ptr " << buffer << " (name='" << mem_name_.at(buffer)
      << "') with " << nbytes << " bytes";
    log.verbose(o.str().c_str());

    int failed = omp_target_memcpy(dev_buffer, buffer, nbytes, 0, 0, target_dev_,
                                   host_dev_);
    if (failed != 0) {
        o.str("");
        o << "OmpManager:  copy of host ptr " << buffer
          << " (name='" << mem_name_.at(buffer)
          << "') with " << nbytes << " bytes to device "
          << target_dev_ << ", failed target memcpy";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }

    #else // ifdef HAVE_OPENMP_TARGET

    o << "OmpManager:  OpenMP target support disabled";
    log.error(o.str().c_str());
    throw std::runtime_error(o.str().c_str());

    #endif // ifdef HAVE_OPENMP_TARGET
}

void OmpManager::update_host(void * buffer, size_t nbytes, std::string const & name) {
    size_t n = mem_size_.count(buffer);
    auto log = toast::Logger::get();
    std::ostringstream o;

    // If the device is the host device, return
    if (device_is_host()) {
        return;
    }

    #ifdef HAVE_OPENMP_TARGET

    if (n == 0) {
        o.str("");
        o << "OmpManager:  host ptr " << buffer
          << " (name='" << name
          << "') is not present- cannot update host";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        size_t nb = mem_size_.at(buffer);
        if (name != mem_name_.at(buffer)) {
            o.str("");
            o << "OmpManager:  on update host, host ptr " << buffer
              << " has name '" << mem_name_.at(buffer) << "' in the table, not '"
              << name << "'";
            log.warning(o.str().c_str());
        }
        if (nb < nbytes) {
            o.str("");
            o << "OmpManager:  on update host, host ptr " << buffer
              << " (name='" << mem_name_.at(buffer)
              << "') has "
              << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    void * dev_buffer = mem_.at(buffer);
    o.str("");
    o << "  OmpManager:  update host ptr " << buffer
      << " (name='" << mem_name_.at(buffer)
      << "') from device ptr "
      << dev_buffer << " with " << nbytes << " bytes";
    log.verbose(o.str().c_str());

    int failed = omp_target_memcpy(buffer, dev_buffer, nbytes, 0, 0, host_dev_,
                                   target_dev_);
    if (failed != 0) {
        o.str("");
        o << "OmpManager:  copy of dev ptr " << dev_buffer
          << " (name='" << mem_name_.at(buffer)
          << "') with " << nbytes << " bytes from device "
          << target_dev_ << ", failed target memcpy";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }

    #else // ifdef HAVE_OPENMP_TARGET

    o << "OmpManager:  OpenMP target support disabled";
    log.error(o.str().c_str());
    throw std::runtime_error(o.str().c_str());

    #endif // ifdef HAVE_OPENMP_TARGET
}

void OmpManager::reset(void * buffer, size_t nbytes, std::string const & name) {
    size_t n = mem_size_.count(buffer);
    auto log = toast::Logger::get();
    std::ostringstream o;

    // If the device is the host device, return
    if (device_is_host()) {
        return;
    }

    #ifdef HAVE_OPENMP_TARGET

    if (n == 0) {
        o.str("");
        o << "OmpManager:  host ptr " << buffer
          << " (name='" << name
          << "') is not present- cannot reset data";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        size_t nb = mem_size_.at(buffer);
        if (name != mem_name_.at(buffer)) {
            o.str("");
            o << "OmpManager:  on reset, host ptr " << buffer
              << " has name '" << mem_name_.at(buffer) << "' in the table, not '"
              << name << "'";
            log.warning(o.str().c_str());
        }
        if (nb < nbytes) {
            o.str("");
            o << "OmpManager:  on reset, host ptr " << buffer
              << " (name='" << mem_name_.at(buffer) << "') has "
              << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    void * dev_buffer = mem_.at(buffer);
    o.str("");
    o << "  OmpManager:  reset device ptr " << dev_buffer
      << " from host ptr " << buffer << " (name='" << mem_name_.at(buffer)
      << "') with " << nbytes << " bytes";
    log.verbose(o.str().c_str());

    # pragma omp target data \
    map(to: nbytes)
    {
        # pragma omp target teams distribute is_device_ptr(dev_buffer) \
        parallel for default(shared)
        for (size_t i = 0; i < nbytes; ++i) {
            ((uint8_t *)dev_buffer)[i] = 0;
        }
    }

    #else // ifdef HAVE_OPENMP_TARGET

    o << "OmpManager:  OpenMP target support disabled";
    log.error(o.str().c_str());
    throw std::runtime_error(o.str().c_str());

    #endif // ifdef HAVE_OPENMP_TARGET
}

int OmpManager::present(void * buffer, size_t nbytes, std::string const & name) {
    auto log = toast::Logger::get();
    std::ostringstream o;

    // If the device is the host device, return
    if (device_is_host()) {
        return 1;
    }

    #ifdef HAVE_OPENMP_TARGET

    size_t n = mem_.count(buffer);
    if (n == 0) {
        return 0;
    } else {
        size_t nb = mem_size_.at(buffer);
        if (name != mem_name_.at(buffer)) {
            o.str("");
            o << "OmpManager:  present, host ptr " << buffer
              << " has name '" << mem_name_.at(buffer) << "' in the table, not '"
              << name << "'";
            log.warning(o.str().c_str());
        }
        if (nb != nbytes) {
            o << "OmpManager:  host ptr " << buffer
              << " (name='" << mem_name_.at(buffer)
              << "') is present, but has " << nb
              << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
        return 1;
    }

    #else // ifdef HAVE_OPENMP_TARGET

    o << "OmpManager:  OpenMP target support disabled";
    log.error(o.str().c_str());
    throw std::runtime_error(o.str().c_str());

    return 0;

    #endif // ifdef HAVE_OPENMP_TARGET
}

void OmpManager::dump() {
    #ifdef HAVE_OPENMP_TARGET
    for (auto & p : mem_size_) {
        void * dev = mem_.at(p.first);
        std::cout << "OmpManager table:  " << mem_name_.at(p.first) << ": "
                  << p.first << " -> "
                  << p.second << " bytes on device " << target_dev_
                  << " at " << dev << std::endl;
    }
    #endif // ifdef HAVE_OPENMP_TARGET
    return;
}

OmpManager::OmpManager() : mem_size_(), mem_() {
    auto log = toast::Logger::get();
    std::ostringstream o;
    host_dev_ = -1;
    #ifdef HAVE_OPENMP_TARGET
    host_dev_ = omp_get_initial_device();
    #endif // ifdef HAVE_OPENMP_TARGET
    target_dev_ = -2;
    node_procs_ = -1;
    node_rank_ = -1;
    pool_ = nullptr;
}

OmpManager::~OmpManager() {
    auto log = toast::Logger::get();
    std::ostringstream o;
    clear();
    if (pool_ != nullptr) {
        delete pool_;
    }
}

void OmpManager::clear() {
    #ifdef HAVE_OPENMP_TARGET
    std::vector <void *> to_clear;
    for (auto & p : mem_) {
        to_clear.push_back(p.first);
    }
    for (auto & p : to_clear) {
        remove(p, mem_size_.at(p));
    }
    #endif // ifdef HAVE_OPENMP_TARGET
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
    }
    if (info.strides[info.ndim - 1] != info.itemsize) {
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "Cannot use python buffers with stride of final dimension != itemsize.";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
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

            #else // ifdef HAVE_OPENMP_TARGET
            return false;

            #endif // ifdef HAVE_OPENMP_TARGET
        },
        R"(
            Return True if TOAST was compiled with OpenMP target offload support.
        )");

    m.def(
        "accel_assign_device",
        [](int node_procs, int node_rank, float mem_gb, bool disabled)
        {
            auto & omgr = OmpManager::get();
            omgr.assign_device(node_procs, node_rank, mem_gb, disabled);
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
            #endif // ifndef HAVE_OPENMP_TARGET

            auto & omgr = OmpManager::get();
            return omgr.get_device();
        },
        R"(
            Return the OpenMP target device assigned to this process.
        )");

    m.def(
        "accel_present", [](py::buffer data, std::string name)
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
            #endif // ifndef HAVE_OPENMP_TARGET

            auto & omgr = OmpManager::get();
            int result = omgr.present(p_host, n_bytes, name);

            if (result == 0) {
                return false;
            } else {
                return true;
            }
        },
        py::arg(
            "data"), py::arg(
            "name"),
        R"(
        Check if the specified array is present on the accelerator device.

        Args:
            data (array):  The data array
            name (str):  The name associated with the data.

        Returns:
            (bool):  True if the data is present, else False.

    )");

    m.def(
        "accel_create", [](py::buffer data, std::string name)
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
            #endif // ifndef HAVE_OPENMP_TARGET

            auto & omgr = OmpManager::get();
            int present = omgr.present(p_host, n_bytes, name);
            if (present == 1) {
                o.str("");
                o << "Data is already present on device, cannot create.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            void * dev_mem = omgr.create(p_host, n_bytes, name);
            return;
        },
        py::arg(
            "data"), py::arg(
            "name"),
        R"(
        Create device copy of the data.

        Args:
            data (array):  The host data.
            name (str):  The name associated with the data.

        Returns:
            None

    )");

    m.def(
        "accel_reset", [](py::buffer data, std::string name)
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
            #endif // ifndef HAVE_OPENMP_TARGET

            auto & omgr = OmpManager::get();
            int present = omgr.present(p_host, n_bytes, name);
            if (present == 0) {
                o.str("");
                o << "Data is not present on device, cannot reset.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            omgr.reset(p_host, n_bytes, name);
            return;
        }, py::arg(
            "data"), py::arg(
            "name"),
        R"(
        Reset device copy of the data to zero.

        This is done directly on the device, without copying zeros.

        Args:
            data (array):  The host data.
            name (str):  The name associated with the data.

        Returns:
            None

    )");

    m.def(
        "accel_update_device", [](py::buffer data, std::string name)
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
            #endif // ifndef HAVE_OPENMP_TARGET

            auto & omgr = OmpManager::get();
            int present = omgr.present(p_host, n_bytes, name);
            if (present == 0) {
                o.str("");
                o << "Data is not present on device, cannot update.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            omgr.update_device(p_host, n_bytes, name);
            return;
        },
        py::arg(
            "data"), py::arg(
            "name"),
        R"(
        Update device copy of the data from the host.

        Args:
            data (array):  The host data.
            name (str):  The name associated with the data.

        Returns:
            None

    )");

    m.def(
        "accel_update_host", [](py::buffer data, std::string name)
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
            #endif // ifndef HAVE_OPENMP_TARGET

            auto & omgr = OmpManager::get();
            int present = omgr.present(p_host, n_bytes, name);
            if (present == 0) {
                o.str("");
                o << "Data is not present on device, cannot update host.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            omgr.update_host(p_host, n_bytes, name);
            return;
        },
        py::arg(
            "data"), py::arg(
            "name"),
        R"(
        Update host copy of the data from the device.

        Args:
            data (array):  The host data.
            name (str):  The name associated with the data.

        Returns:
            None

    )");

    m.def(
        "accel_delete", [](py::buffer data, std::string name)
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
            #endif // ifndef HAVE_OPENMP_TARGET

            auto & omgr = OmpManager::get();
            int present = omgr.present(p_host, n_bytes, name);
            if (present == 0) {
                o.str("");
                o << "Data is not present on device, cannot delete.";
                log.error(o.str().c_str());
                throw std::runtime_error(o.str().c_str());
            }

            omgr.remove(p_host, n_bytes, name);
            return;
        },
        py::arg(
            "data"), py::arg(
            "name"),
        R"(
        Delete the device copy of the data.

        Args:
            data (array):  The host data.
            name (str):  The name associated with the data.

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
            std::vector <int64_t> temp_shape(2);

            double * raw = extract_buffer <double> (
                data, "data", 2, temp_shape, {-1, -1}
            );
            int64_t n_det = temp_shape[0];
            int64_t n_samp = temp_shape[1];

            auto & omgr = OmpManager::get();
            int dev = omgr.get_device();
            bool offload = !omgr.device_is_host();

            void * dev_raw = omgr.device_ptr(raw);

            #ifdef HAVE_OPENMP_TARGET
            # pragma omp target data
            {
                # pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(dev_raw)
                for (int64_t i = 0; i < n_det; i++) {
                    for (int64_t j = 0; j < n_samp; j++) {
                        raw[i * n_samp + j] *= 2.0;
                    }
                }
            }
            #endif // ifdef HAVE_OPENMP_TARGET
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
            bool offload = !omgr.device_is_host();
            void * dev_raw = omgr.device_ptr(raw);

            #ifdef HAVE_OPENMP_TARGET
            # pragma omp target data
            {
                # pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(dev_raw)
                for (size_t i = 0; i < n_det; i++) {
                    for (size_t j = 0; j < n_samp; j++) {
                        raw[i * n_samp + j] *= 2.0;
                    }
                }
            }
            #endif // ifdef HAVE_OPENMP_TARGET
            return;
        });
}
