// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef LIBTOAST_INTERVALS_HPP
#define LIBTOAST_INTERVALS_HPP

#include <common.hpp>

struct Interval {
    double start;
    double stop;
    int64_t first;
    int64_t last;
};

#endif // ifndef LIBTOAST_INTERVALS_HPP
