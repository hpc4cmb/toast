/*
Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_SIM_MAP_HPP
#define TOAST_SIM_MAP_HPP


namespace toast { namespace sim_map {


    template < typename T >
    void scan_map (
        long *submap, long subnpix, double *weights, size_t nmap, long *subpix,
        T *map, double *tod, size_t nsamp ) {

        #pragma omp for schedule(static)
        for ( long i=0; i<nsamp; ++i ) {
            tod[i] = 0;
            long offset = (submap[i]*subnpix+subpix[i]) * nmap;
            long woffset = i * nmap;
            for ( int imap=0; imap<nmap; ++imap ) {
                tod[i] += map[offset++] * weights[woffset++];
            }
        }

        return;
    };

} }

#endif
