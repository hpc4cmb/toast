/*
   Copyright (c) 2015-2018 by the parties listed in the AUTHORS file. All rights
      reserved.  Use of this source code is governed by a BSD-style license that can be
      found in the LICENSE file.
 */

#ifndef TOAST_MATH_SF_HPP
#define TOAST_MATH_SF_HPP


namespace toast {
void vsin(int n, double const * ang, double * sinout);
void vcos(int n, double const * ang, double * cosout);
void vsincos(int n, double const * ang, double * sinout, double * cosout);
void vatan2(int n, double const * y, double const * x, double * ang);
void vsqrt(int n, double const * in, double * out);
void vrsqrt(int n, double const * in, double * out);
void vexp(int n, double const * in, double * out);
void vlog(int n, double const * in, double * out);

void vfast_sin(int n, double const * ang, double * sinout);
void vfast_cos(int n, double const * ang, double * cosout);
void vfast_sincos(int n, double const * ang, double * sinout, double * cosout);
void vfast_atan2(int n, double const * y, double const * x, double * ang);
void vfast_sqrt(int n, double const * in, double * out);
void vfast_rsqrt(int n, double const * in, double * out);
void vfast_exp(int n, double const * in, double * out);
void vfast_log(int n, double const * in, double * out);
void vfast_erfinv(int n, double const * in, double * out);
}

#endif // ifndef TOAST_SF_HPP
