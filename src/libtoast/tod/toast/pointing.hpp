/*
Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_POINTING_HPP
#define TOAST_POINTING_HPP


namespace toast { namespace pointing {

    void healpix_matrix ( toast::healpix::pixels const & hpix, 
        bool nest, double eps, double cal, std::string const & mode, size_t n,
        double const * pdata, double const * hwpang, uint8_t const * flags,
        int64_t * pixels, double * weights );


} }

#endif

