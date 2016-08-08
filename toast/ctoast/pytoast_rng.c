/*
Copyright (c) 2016 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


/* Interface function for counter-based random number generation */
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include <threefry.h>
#include <pytoast.h>


/*
 * Converts unsigned 64-bit integer to double.
 * Output is as dense as possible in (0,1), never 0.0.
 */
double pytoast_u01(uint64_t in) {
    double factor = 1. / (UINT64_MAX + 1.);
    double halffactor = 0.5 * factor;
    return (in * factor + halffactor);
}

/*
 * Converts unsigned 64-bit integer to double.
 * Output is as dense as possible in (-1,1), never 0.0.
 */
double pytoast_uneg11(uint64_t in) {
    double factor = 1. / (INT64_MAX + 1.);
    double halffactor = 0.5 * factor;
    return (((int64_t)in) * factor + halffactor);
}

/*
 * Returns gaussian random variables generated with threefry2x64 and transformed with Box-Muller.
 *
 * size: even number of variables to return
 * offset: variables are stored starting from rand_array[offset]
 * counter1 and counter2: first and second 64-bit component of the counter
 * rand_array: array of size at least [offset+size] where the random variables are written
 */
void pytoast_generate_grv(uint64_t size, uint64_t offset, uint64_t counter1, uint64_t counter2, uint64_t key1, uint64_t key2, double* rand_array) {
    uint64_t i;
    threefry2x64_ctr_t rand;

    /* Box-Muller transform variables */
    double twoPI = 2.0 * 3.1415926535897932;

    threefry2x64_key_t key={{key1, key2}};
    threefry2x64_ctr_t ctr={{counter1, counter2}};

    for (i = 0; i < size; i++) {
        rand = threefry2x64(ctr, key);
        rand_array[i+offset] = sqrt(-2.0 * log(pytoast_u01(rand.v[0]))) * cos(twoPI * pytoast_u01(rand.v[1]));
        ctr.v[1]++;
    }
    return;
}

/*
 * Returns uniform random variables in (0,1) \ {0} generated with threefry2x64.
 *
 * size: even number of variables to return
 * offset: variables are stored starting from rand_array[offset]
 * counter1 and counter2: first and second 64-bit component of the counter
 * rand_array: array of size at least [offset+size] where the random variables are written
 */
void pytoast_generate_neg11rv(uint64_t size, uint64_t offset, uint64_t counter1, uint64_t counter2, uint64_t key1, uint64_t key2, double* rand_array) {
    uint64_t i;
    threefry2x64_ctr_t rand;

    threefry2x64_key_t key={{key1, key2}};
    threefry2x64_ctr_t ctr={{counter1, counter2}};

    for (i = 0; i < size; i++) {
        rand = threefry2x64(ctr, key);
        rand_array[i+offset] = pytoast_uneg11(rand.v[0]);
        ctr.v[1]++;
    }
    return;
}

/*
 * Returns uniform random variables in (-1,1) \ {0} generated with threefry2x64.
 *
 * size: even number of variables to return
 * offset: variables are stored starting from rand_array[offset]
 * counter1 and counter2: first and second 64-bit component of the counter
 * rand_array: array of size at least [offset+size] where the random variables are written
 */
void pytoast_generate_01rv(uint64_t size, uint64_t offset, uint64_t counter1, uint64_t counter2, uint64_t key1, uint64_t key2, double* rand_array) {
    uint64_t i;
    threefry2x64_ctr_t rand;

    threefry2x64_key_t key={{key1, key2}};
    threefry2x64_ctr_t ctr={{counter1, counter2}};

    for (i = 0; i < size; i++) {
        rand = threefry2x64(ctr, key);
        rand_array[i+offset] = pytoast_u01(rand.v[0]);
        ctr.v[1]++;
    }
    return;
}


/*
 * Returns uniform natural random variables (unsigned 64-bit integers) generated with threefry2x64.
 *
 * size: even number of variables to return
 * offset: variables are stored starting from rand_array[offset]
 * counter1 and counter2: first and second 64-bit component of the counter
 * rand_array: array of size at least [offset+size] where the random variables are written
 */
void pytoast_generate_uint64rv(uint64_t size, uint64_t offset, uint64_t counter1, uint64_t counter2, uint64_t key1, uint64_t key2, uint64_t* rand_array) {
    uint64_t i;
    threefry2x64_ctr_t rand;

    threefry2x64_key_t key={{key1, key2}};
    threefry2x64_ctr_t ctr={{counter1, counter2}};

    for (i = 0; i < size; i++) {
        rand = threefry2x64(ctr, key);
        rand_array[i+offset] = rand.v[0];
        ctr.v[1]++;
    }
    return;
}

