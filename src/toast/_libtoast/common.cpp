
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <common.hpp>

// FIXME: we could use configure checks to detect whether we are on a 64bit
// system and whether "l" and "L" are equivalent to "q" and "Q".

template <>
std::vector <char> align_format <int8_t> () {
    return std::vector <char> ({'b'});
}

template <>
std::vector <char> align_format <int16_t> () {
    return std::vector <char> ({'h'});
}

template <>
std::vector <char> align_format <int32_t> () {
    return std::vector <char> ({'i'});
}

template <>
std::vector <char> align_format <int64_t> () {
    return std::vector <char> ({'q', 'l'});
}

template <>
std::vector <char> align_format <uint8_t> () {
    return std::vector <char> ({'B'});
}

template <>
std::vector <char> align_format <uint16_t> () {
    return std::vector <char> ({'H'});
}

template <>
std::vector <char> align_format <uint32_t> () {
    return std::vector <char> ({'I'});
}

template <>
std::vector <char> align_format <uint64_t> () {
    return std::vector <char> ({'Q', 'L'});
}

template <>
std::vector <char> align_format <float> () {
    return std::vector <char> ({'f'});
}

template <>
std::vector <char> align_format <double> () {
    return std::vector <char> ({'d'});
}

FakeMemPool & FakeMemPool::get() {
    static FakeMemPool instance;
    return instance;
}

void * FakeMemPool::create(void * buffer, size_t nbytes) {
    size_t n = mem_size.count(buffer);
    std::ostringstream o;
    auto log = toast::Logger::get();
    if (n != 0) {
        o << "FakeMemPool:  on create, host ptr " << buffer
          << " with " << nbytes << " bytes is already present "
          << "with " << mem_size.at(buffer) << " bytes";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }

    // Add to the map
    o.str("");
    o << "FakeMemPool:  creating entry for host ptr "
      << buffer << " with " << nbytes << " bytes";
    log.verbose(o.str().c_str());
    mem_size[buffer] = nbytes;
    mem[buffer] = malloc(nbytes);
    if (mem.at(buffer) == NULL) {
        o.str("");
        o << "FakeMemPool:  on create, host ptr " << buffer
          << " with " << nbytes << " bytes, device allocation failed";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    return mem.at(buffer);
}

void FakeMemPool::remove(void * buffer, size_t nbytes) {
    size_t n = mem_size.count(buffer);
    auto log = toast::Logger::get();
    std::ostringstream o;
    if (n == 0) {
        o.str("");
        o << "FakeMemPool:  host ptr " << buffer
          << " is not present- cannot delete";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        size_t nb = mem_size.at(buffer);
        if (nb != nbytes) {
            o.str("");
            o << "FakeMemPool:  on delete, host ptr " << buffer << " has "
              << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    o.str("");
    o << "FakeMemPool:  removing entry for host ptr "
      << buffer << " with " << nbytes << " bytes";
    log.verbose(o.str().c_str());
    mem_size.erase(buffer);
    free(mem.at(buffer));
    mem.erase(buffer);
}

void * FakeMemPool::copyin(void * buffer, size_t nbytes) {
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
            o << "FakeMemPool:  on copyin, host ptr " << buffer << " has "
              << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    ptr = mem.at(buffer);
    o.str("");
    o << "FakeMemPool:  copy in host ptr "
      << buffer << " with " << nbytes << " bytes to device "
      << ptr;
    log.verbose(o.str().c_str());
    void * temp = memcpy(ptr, buffer, nbytes);
    return mem.at(buffer);
}

void FakeMemPool::copyout(void * buffer, size_t nbytes) {
    size_t n = mem_size.count(buffer);
    auto log = toast::Logger::get();
    std::ostringstream o;
    size_t nb;
    if (n == 0) {
        o.str("");
        o << "FakeMemPool:  host ptr " << buffer
          << " is not present- cannot copy out";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        nb = mem_size.at(buffer);
        if (nb < nbytes) {
            o.str("");
            o << "FakeMemPool:  on copyout, host ptr " << buffer << " has "
              << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    void * ptr = mem.at(buffer);
    o.str("");
    o << "FakeMemPool:  copy out host ptr "
      << buffer << " with " << nbytes << " bytes from device "
      << ptr;
    void * temp = memcpy(buffer, ptr, nbytes);

    // Even if we copyout a portion of the buffer, remove the full thing.
    remove(buffer, nb);
}

void FakeMemPool::update_device(void * buffer, size_t nbytes) {
    size_t n = mem_size.count(buffer);
    auto log = toast::Logger::get();
    std::ostringstream o;
    if (n == 0) {
        o.str("");
        o << "FakeMemPool:  host ptr " << buffer
          << " is not present- cannot update device";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        size_t nb = mem_size.at(buffer);
        if (nb < nbytes) {
            o.str("");
            o << "FakeMemPool:  on update device, host ptr " << buffer << " has "
              << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    void * dev = mem.at(buffer);
    o.str("");
    o << "FakeMemPool:  update device from host ptr "
      << buffer << " with " << nbytes << " to " << dev;
    void * temp = memcpy(dev, buffer, nbytes);
}

void FakeMemPool::update_self(void * buffer, size_t nbytes) {
    size_t n = mem_size.count(buffer);
    auto log = toast::Logger::get();
    std::ostringstream o;
    if (n == 0) {
        o.str("");
        o << "FakeMemPool:  host ptr " << buffer
          << " is not present- cannot update host";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    } else {
        size_t nb = mem_size.at(buffer);
        if (nb < nbytes) {
            o.str("");
            o << "FakeMemPool:  on update self, host ptr " << buffer << " has "
              << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
    }
    void * dev = mem.at(buffer);
    o.str("");
    o << "FakeMemPool:  update host ptr "
      << buffer << " with " << nbytes << " from " << dev;
    void * temp = memcpy(buffer, dev, nbytes);
}

// Use int instead of bool to match the acc_is_present API
int FakeMemPool::present(void * buffer, size_t nbytes) {
    size_t n = mem_size.count(buffer);
    if (n == 0) {
        return 0;
    } else {
        size_t nb = mem_size.at(buffer);
        if (nb != nbytes) {
            auto log = toast::Logger::get();
            std::ostringstream o;
            o << "FakeMemPool:  host ptr " << buffer << " is present"
              << ", but has " << nb << " bytes instead of " << nbytes;
            log.error(o.str().c_str());
            throw std::runtime_error(o.str().c_str());
        }
        return 1;
    }
}

void * FakeMemPool::device_ptr(void * buffer) {
    auto log = toast::Logger::get();
    std::ostringstream o;
    size_t n = mem.count(buffer);
    if (n == 0) {
        o.str("");
        o << "FakeMemPool:  host ptr " << buffer
          << " is not present- cannot get device pointer";
        log.error(o.str().c_str());
        throw std::runtime_error(o.str().c_str());
    }
    return mem.at(buffer);
}

void FakeMemPool::dump() {
    for (auto & p : mem_size) {
        void * dev = mem.at(p.first);
        std::cout << "FakeMemPool table:  " << p.first << ": "
                  << p.second << " bytes on dev at " << dev << std::endl;
    }
    return;
}

FakeMemPool::FakeMemPool() : mem_size(), mem() {}

FakeMemPool::~FakeMemPool() {
    for (auto & p : mem) {
        free(p.second);
    }
    mem_size.clear();
    mem.clear();
}
