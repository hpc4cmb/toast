/*
test the implementation of quaternion arrays.
*/

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include <pytoast.h>

int main(int argc, char const *argv[])
{
    int i;
    int n = 11, offset = 1;
    uint64_t counter1 = 1357111317, counter2 = 888118218888;
    uint64_t key1 = 3405692589, key2 = 3131965165;

    double * array_f64 = malloc( (n + offset) * sizeof(double) );
    uint64_t * array_uint64 = malloc( (n + offset) * sizeof(uint64_t) );

    pytoast_generate_grv(n, offset, counter1, counter2, key1, key2, array_f64);

    printf("x_gaussian = [");
    for (i = 0; i < n; ++i) printf("%f, ", array_f64[offset+i]);
    printf("]\n");

    pytoast_generate_neg11rv(n, offset, counter1, counter2, key1, key2, array_f64);

    printf("x_m11 = [");
    for (i = 0; i < n; ++i) printf("%f, ", array_f64[offset+i]);
    printf("]\n");

    pytoast_generate_01rv(n, offset, counter1, counter2, key1, key2, array_f64);

    printf("x_01 = [");
    for (i = 0; i < n; ++i) printf("%f, ", array_f64[offset+i]);
    printf("]\n");

    pytoast_generate_uint64rv(n, offset, counter1, counter2, key1, key2, array_uint64);

    printf("x_uint64 = [");
    for (i = 0; i < n; ++i) printf("%lu, ", (long unsigned)array_uint64[offset+i]);
    printf("]\n");

    counter1 = 0;
    counter2 = 0;
    key1 = 0;
    key2 = 0;

    pytoast_generate_grv(n, offset, counter1, counter2, key1, key2, array_f64);

    printf("x_gaussian = [");
    for (i = 0; i < n; ++i) printf("%f, ", array_f64[offset+i]);
    printf("]\n");

    pytoast_generate_neg11rv(n, offset, counter1, counter2, key1, key2, array_f64);

    printf("x_m11 = [");
    for (i = 0; i < n; ++i) printf("%f, ", array_f64[offset+i]);
    printf("]\n");

    pytoast_generate_01rv(n, offset, counter1, counter2, key1, key2, array_f64);

    printf("x_01 = [");
    for (i = 0; i < n; ++i) printf("%f, ", array_f64[offset+i]);
    printf("]\n");

    pytoast_generate_uint64rv(n, offset, counter1, counter2, key1, key2, array_uint64);

    printf("x_uint64 = [");
    for (i = 0; i < n; ++i) printf("%lu, ", (long unsigned)array_uint64[offset+i]);
    printf("]\n");

    free(array_f64);
    free(array_uint64);

    return 0;
}
