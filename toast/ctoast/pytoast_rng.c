/*
Copyright (c) 2016 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


/* Interface function for counter-based random number generation */

#include <Random123/threefry.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <limits.h>

/* We might consider implementing a version returning 4 random number at a time for better efficiency */

void generate_cbrn(const unsigned long int size, const unsigned long int offset, const unsigned long int counter1, const unsigned long int counter2, double* rand_array) {
    int i;
    threefry2x64_ctr_t rand;

    /* Box-Muller transform variables */
    const double epsilon = DBL_MIN;
    const double two_pi = 2.0*3.14159265358979323846;
    double* x, y;
    double r;

    threefry2x64_key_t key={{0, 0}};
    threefry2x64_ctr_t ctr={{counter1, counter2}};

    /* Use a union to avoid strict aliasing issues. */
    /* 
    enum { int32s_per_counter = sizeof(ctr)/sizeof(int32_t) };
    assert( int32s_per_counter%2 == 0 );
    */

    for (i = 0; i < size; i+=2) {

        /* Use a union to avoid strict aliasing issues. */
        /*
        union{
            threefry2x64_ctr_t ct;
            int32_t i32[int32s_per_counter];
        }u;
        */


        do
        {
            ctr.v[0]++;
            rand = threefry2x64(ctr, key);
            /* Change the conversion for anti-aliasing reasons */
            rand_array[i+offset] = rand.v[0] / ULLONG_MAX;
            rand_array[i+1+offset] = rand.v[1] / ULLONG_MAX;
        }
        while ( (rand_array[i+offset] <= epsilon) && (rand_array[i+1+offset] <= epsilon) );

        sincos(two_pi*rand_array[i+1+offset], x, y);

        r = sqrt(-2. * log(rand_array[i+offset]));

        rand_array[i+offset] = x*r;
        rand_array[i+1+offset] = y*r;

    }
}
