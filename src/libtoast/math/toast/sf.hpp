/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_SF_HPP
#define TOAST_SF_HPP


namespace toast { namespace sf {

    void sin ( int n, double const * ang, double * sinout );
    void cos ( int n, double const * ang, double * cosout );
    void sincos ( int n, double const * ang, double * sinout, double * cosout );
    void atan2 ( int n, double const * y, double const * x, double * ang );
    void sqrt ( int n, double const * in, double * out );
    void rsqrt ( int n, double const * in, double * out );
    void exp ( int n, double const * in, double * out );
    void log ( int n, double const * in, double * out );

    void fast_sin ( int n, double const * ang, double * sinout );
    void fast_cos ( int n, double const * ang, double * cosout );
    void fast_sincos ( int n, double const * ang, double * sinout, double * cosout );
    void fast_atan2 ( int n, double const * y, double const * x, double * ang );
    void fast_sqrt ( int n, double const * in, double * out );
    void fast_rsqrt ( int n, double const * in, double * out );
    void fast_exp ( int n, double const * in, double * out );
    void fast_log ( int n, double const * in, double * out );
    void fast_erfinv ( int n, double const * in, double * out );


} }

#endif

