// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef LIBTOAST_QARRAY_HPP
#define LIBTOAST_QARRAY_HPP

#include <common.hpp>

// This header defines low-level quaternion functions.

#pragma omp declare target

void qa_normalize_inplace(size_t n, double * q);

void qa_rotate(double const * q_in, double const * v_in, double * v_out);

void qa_mult(double const * p, double const * q, double * r);

#pragma omp end declare target

// FIXME:  move remaining functions from libtoast here and in qarray_core.cpp

#endif // ifndef LIBTOAST_QARRAY_HPP
