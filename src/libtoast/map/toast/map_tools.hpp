/*
Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_MAP_TOOLS_HPP
#define TOAST_MAP_TOOLS_HPP


namespace toast { namespace map_tools {

void fast_scanning32(double * toi, int64_t const nsamp,
                     int64_t const * pixels,
                     double const * weights,
                     int64_t const nweight,
                     float const * bmap);

} }

#endif
