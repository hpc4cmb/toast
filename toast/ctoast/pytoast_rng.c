/*
Copyright (c) 2016 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


/* Interface function for counter-based random number generation */

#include <stdint.h>
#include <math.h>

#include <threefry.h>
#include <pytoast.h>

/* We might consider implementing a version returning 4 random number per rng call (using threefry4x64) for better efficiency */

/* 
 * Some versions of OS X do not have sincos in the standard math library.
 * Also, some versions of the math library on Linux requre GNU extensions
 * to be enabled to access this function.  Rather than get into complicated
 * logic to detect OS features, we just define our own function.  If this
 * really becomes a performance issue, we can investigate at that time.
 */
void rng_sincos(double x, double *s, double *c) {
    *s = sin(x);
    *c = cos(x);
}

/*
 * Converts unsigned 64-bit integer to double.
 * Output is as dense as possible in (0,1), never 0.0.
 */
double u01(uint64_t in) {
    double factor = 1. / (UINT64_MAX + 1.);
    double halffactor = 0.5 * factor;
    return (in * factor + halffactor);
}

/*
 * Converts unsigned 64-bit integer to double.
 * Output is as dense as possible in (-1,1), never 0.0.
 */
double uneg11(uint64_t in) {
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
void generate_grv(uint64_t size, uint64_t offset, uint64_t counter1, uint64_t counter2, uint64_t key1, uint64_t key2, double* rand_array) {
    int i;
    threefry2x64_ctr_t rand;

    /* Box-Muller transform variables */
    const double PI = 3.1415926535897932;
    double x, y, r;

    threefry2x64_key_t key={{key1, key2}};
    threefry2x64_ctr_t ctr={{counter1, counter2}};

    for (i = 0; i < (size - (size%2)); i+=2) {
        rand = threefry2x64(ctr, key);
        rand_array[i+offset] = rand.v[0];
        rand_array[i+1+offset] = rand.v[1];

        rng_sincos(PI*uneg11(rand_array[i+offset]), &x, &y);
        r = sqrt(-2. * log(u01(rand_array[i+1+offset])));

        rand_array[i+offset] = x * r;
        rand_array[i+1+offset] = y * r;

        ctr.v[1]++;
    }
}

/*
 * Returns uniform random variables in (0,1) \ {0} generated with threefry2x64.
 *
 * size: even number of variables to return
 * offset: variables are stored starting from rand_array[offset]
 * counter1 and counter2: first and second 64-bit component of the counter
 * rand_array: array of size at least [offset+size] where the random variables are written
 */
void generate_neg11rv(uint64_t size, uint64_t offset, uint64_t counter1, uint64_t counter2, uint64_t key1, uint64_t key2, double* rand_array) {
    int i;
    threefry2x64_ctr_t rand;

    threefry2x64_key_t key={{key1, key2}};
    threefry2x64_ctr_t ctr={{counter1, counter2}};

    for (i = 0; i < (size - (size%2)); i+=2) {
        rand = threefry2x64(ctr, key);
        rand_array[i+offset] = uneg11(rand.v[0]);
        rand_array[i+1+offset] = uneg11(rand.v[1]);

        ctr.v[1]++;
    }
}

/*
 * Returns uniform random variables in (-1,1) \ {0} generated with threefry2x64.
 *
 * size: even number of variables to return
 * offset: variables are stored starting from rand_array[offset]
 * counter1 and counter2: first and second 64-bit component of the counter
 * rand_array: array of size at least [offset+size] where the random variables are written
 */
void generate_01rv(uint64_t size, uint64_t offset, uint64_t counter1, uint64_t counter2, uint64_t key1, uint64_t key2, double* rand_array) {
    int i;
    threefry2x64_ctr_t rand;

    threefry2x64_key_t key={{key1, key2}};
    threefry2x64_ctr_t ctr={{counter1, counter2}};

    for (i = 0; i < (size - (size%2)); i+=2) {
        rand = threefry2x64(ctr, key);
        rand_array[i+offset] = u01(rand.v[0]);
        rand_array[i+1+offset] = u01(rand.v[1]);

        ctr.v[1]++;
    }
}


/*
 * Returns uniform natural random variables (unsigned 64-bit integers) generated with threefry2x64.
 *
 * size: even number of variables to return
 * offset: variables are stored starting from rand_array[offset]
 * counter1 and counter2: first and second 64-bit component of the counter
 * rand_array: array of size at least [offset+size] where the random variables are written
 */
void generate_uint64rv(uint64_t size, uint64_t offset, uint64_t counter1, uint64_t counter2, uint64_t key1, uint64_t key2, uint64_t* rand_array) {
    int i;
    threefry2x64_ctr_t rand;

    threefry2x64_key_t key={{key1, key2}};
    threefry2x64_ctr_t ctr={{counter1, counter2}};

    for (i = 0; i < (size - (size%2)); i+=2) {
        rand = threefry2x64(ctr, key);
        rand_array[i+offset] = rand.v[0];
        rand_array[i+1+offset] = rand.v[1];

        ctr.v[1]++;
    }
}

