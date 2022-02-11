// Copyright (c) 2015-2021 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef LIBTOAST_INTERVALS_HPP
#define LIBTOAST_INTERVALS_HPP

#include <common.hpp>

class Interval {
    public:
        Interval();
        Interval(
            double start_time,
            double stop_time,
            int64_t first_samp,
            int64_t last_samp
        );
        double start;
        double stop;
        int64_t first;
        int64_t last;
};

using IntervalList = std::vector <Interval>;

PYBIND11_MAKE_OPAQUE(IntervalList);

#endif
