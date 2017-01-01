/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_MATH_HPP
#define TOAST_MATH_HPP


#include <limits>
#include <exception>
#include <memory>
#include <vector>
#include <map>

#include <cstdlib>
#include <cstdint>


namespace toast {

    // General Constants

    // Static String Lengths
    static size_t const STRLEN = 256;
    static size_t const BIGSTRLEN = 4096;

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

    class exception : public std::exception {

        public:
            exception ( const char * msg, const char * file, int line );
            ~exception ( ) throw ();
            const char* what() const throw();

        private:  
            char msg_[BIGSTRLEN];

    };  

    typedef void (*TOAST_EXCEPTION_HANDLER) ( toast::exception & e );

    #define TOAST_THROW(msg) \
    throw toast::exception ( msg, __FILE__, __LINE__ )

    #define TOAST_TRY \
    try {

    #define TOAST_CATCH \
    } catch ( toast::exception & e ) { \
        std::cerr << e.what() << std::endl; \
        toast::cleanup(); \
        throw; \
    }

    #define TOAST_CATCH_CUSTOM(handler) \
    } catch ( toast::exception & e ) { \
        (*handler) ( e ); \
    }

}

#include <toast/memory.hpp>
#include <toast/sf.hpp>
#include <toast/rng.hpp>
#include <toast/qarray.hpp>
#include <toast/fft.hpp>
#include <toast/healpix.hpp>

#endif

