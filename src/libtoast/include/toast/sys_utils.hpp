
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_SYS_UTILS_HPP
#define TOAST_SYS_UTILS_HPP

#include <cstddef>
#include <cmath>

#include <iostream>
#include <chrono>
#include <memory>
#include <map>
#include <vector>


namespace toast {

// Constants

// PI
static double const PI = 3.14159265358979323846;

// PI/2
static double const PI_2 = 1.57079632679489661923;

// PI/4
static double const PI_4 = 0.78539816339744830962;

// 1/PI
static double const INV_PI = 0.31830988618379067154;

// 1/(2*PI)
static double const INV_TWOPI = 0.15915494309189533577;

// 2/PI
static double const TWOINVPI = 0.63661977236758134308;

// 2/3
static double const TWOTHIRDS = 0.66666666666666666667;

// 2*PI
static double const TWOPI = 6.28318530717958647693;

// 1/sqrt(2)
static double const INVSQRTTWO = 0.70710678118654752440;

// tan(PI/12)
static double const TANTWELFTHPI = 0.26794919243112270647;

// tan(PI/6)
static double const TANSIXTHPI = 0.57735026918962576451;

// PI/6
static double const SIXTHPI = 0.52359877559829887308;

// 3*PI/2
static double const THREEPI_2 = 4.71238898038468985769;

// Degrees to Radians
static double const DEG2RAD = 1.74532925199432957692e-2;

// Exception handling

class Exception : public std::exception {
    public:

        Exception(const char * msg, const char * file, int line);
        ~Exception() throw();
        const char * what() const throw();

    private:

        // use C strings here for passing to what()
        static size_t const msg_len_ = 1024;
        char msg_[msg_len_];
};

typedef void (* TOAST_EXCEPTION_HANDLER) (toast::Exception & e);

// Helper macro for throwing a toast exception
#define TOAST_THROW(msg) \
    throw toast::Exception(msg, __FILE__, __LINE__)

class Timer {
    // Simple timer class that tracks elapsed seconds and number of times
    // it was started.

    public:

        typedef std::chrono::high_resolution_clock::time_point time_point;
        typedef std::shared_ptr <Timer> pshr;
        typedef std::unique_ptr <Timer> puniq;

        Timer();
        void start();
        void stop();
        void clear();
        double seconds() const;
        void report(char const * message);
        bool is_running() const;

    private:

        double total_;
        time_point start_;
        time_point stop_;
        bool running_;
        size_t calls_;
};


class GlobalTimers {
    // Singleton registry of global timers that can be accessed from anywhere.

    public:

        // Singleton access
        static GlobalTimers & get();

        std::vector <std::string> names() const;
        void start(std::string const & name);
        void stop(std::string const & name);
        void clear(std::string const & name);
        double seconds(std::string const & name) const;
        bool is_running(std::string const & name) const;

        void stop_all();
        void clear_all();

        void report();

    private:

        // This class is a singleton- constructor is private.
        GlobalTimers();

        // The timer data
        std::map <std::string, Timer> data;
};


enum class log_level {
    none     = 0, ///< Undefined
    debug    = 1, ///< Debug
    info     = 2, ///< Info
    warning  = 3, ///< Warning
    error    = 4, ///< Error
    critical = 5  ///< Critical
};


class Logger {
    // Global logger singleton.

    public:

        // Singleton access
        static Logger & get();

        void debug(char const * msg);
        void info(char const * msg);
        void warning(char const * msg);
        void error(char const * msg);
        void critical(char const * msg);

    private:

        // This class is a singleton- constructor is private.
        Logger();
        void check_level();

        log_level level_;
        std::string prefix_;
};

// Aligned memory allocation helpers.

// Byte alignment for SIMD.  This should work for all modern systems.
static size_t const SIMD_ALIGN = 64;

// Low-level C aligned malloc / free.
void * aligned_alloc(size_t size, size_t align);
void aligned_free(void * ptr);

// Check for alignment of a pointer
template <typename T>
bool is_aligned(T * ptr) {
    return !(reinterpret_cast <uintptr_t> (ptr) % SIMD_ALIGN);
}

// Allocator that can be used with STL containers.

template <typename T>
class AlignedAllocator {
    // Custom allocator based on example in:
    // The C++ Standard Library - A Tutorial and Reference
    // by Nicolai M. Josuttis, Addison-Wesley, 1999

    public:

        // type definitions
        typedef T value_type;
        typedef T * pointer;
        typedef T const * const_pointer;
        typedef T & reference;
        typedef T const & const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        // rebind allocator to type U
        template <typename U>
        struct rebind {
            typedef AlignedAllocator <U> other;
        };

        // return address of values
        pointer address(reference value) const {
            return &value;
        }

        const_pointer address(const_reference value) const {
            return &value;
        }

        AlignedAllocator() throw() {}

        AlignedAllocator(AlignedAllocator const &) throw() {}

        template <typename U>
        AlignedAllocator(AlignedAllocator <U> const &) throw() {}

        ~AlignedAllocator() throw() {}

        // return maximum number of elements that can be allocated
        size_type max_size() const throw() {
            return std::numeric_limits <std::size_t>::max() / sizeof(T);
        }

        // allocate but don't initialize num elements of type T
        pointer allocate(size_type const num, const void * hint = 0) {
            pointer align_ptr =
                static_cast <pointer> (aligned_alloc(num * sizeof(T),
                                                     SIMD_ALIGN));

            return align_ptr;
        }

        // initialize elements of allocated storage p with value value
        void construct(pointer p, T const & value) {
            // initialize memory with placement new
            new (static_cast <void *> (p)) T(value);
        }

        // destroy elements of initialized storage p
        void destroy(pointer p) {
            // destroy objects by calling their destructor
            p->~T();
        }

        // deallocate storage p of deleted elements
        void deallocate(pointer p, size_type num) {
            aligned_free(static_cast <void *> (p));
        }
};

// return that all specializations of this allocator are interchangeable
template <typename T1, class T2>
bool operator==(AlignedAllocator <T1> const &,
                AlignedAllocator <T2> const &) throw() {
    return true;
}

template <typename T1, class T2>
bool operator!=(AlignedAllocator <T1> const &,
                AlignedAllocator <T2> const &) throw() {
    return false;
}

// Helper alias for std::vector of a type with a AlignedAllocator for that
// type.
template <typename T>
using AlignedVector = std::vector <T, toast::AlignedAllocator <T> >;

}


#endif // ifndef TOAST_UTILS_HPP
