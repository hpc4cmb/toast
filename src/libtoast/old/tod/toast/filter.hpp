/*
Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_FILTER_HPP
#define TOAST_FILTER_HPP


namespace toast { namespace filter {

void polyfilter(
    const long order, double **signals, uint8_t *flags,
    const size_t n, const size_t nsignal,
    const long *starts, const long *stops, const size_t nscan );

} }

#endif
