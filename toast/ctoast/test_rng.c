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
    int n = 11, offset = 2;
    uint64_t counter1 = 1357111317, counter2 = 888118218888;
    uint64_t key1 = 0xcafebead, key2 = 0xbaadfeed;
    if(argc>1) counter1 = atoi(argv[1]);
    if(argc>2) counter2 = atoi(argv[2]);
    if(argc>3) key1 = atoi(argv[3]);
    if(argc>4) key2 = atoi(argv[4]);
    double * array_f64 = malloc( (n + 2 * offset) * sizeof(double) );
    uint64_t * array_uint64 = malloc( (n + 2 * offset) * sizeof(uint64_t) );

    generate_grv(n, offset, counter1, counter2, key1, key2, array_f64);

    printf("x_gaussian = [ %f ", array_f64[0]);
    for (i = 1; i < (n + 2 * offset); ++i) printf(", %f ", array_f64[i]);
    printf("];\n");


    generate_neg11rv(n, offset, counter1, counter2, key1, key2, array_f64);

    printf("x_m11 = [ %f ", array_f64[0]);
    for (i = 1; i < (n + 2 * offset); ++i) printf(", %f ", array_f64[i]);
    printf("];\n");

 
    generate_01rv(n, offset, counter1, counter2, key1, key2, array_f64);

    printf("x_01 = [ %f ", array_f64[0]);
    for (i = 1; i < (n + 2 * offset); ++i) printf(", %f ", array_f64[i]);
    printf("];\n");


    generate_uint64rv(n, offset, counter1, counter2, key1, key2, array_uint64);

    printf("x_uint64 = [ %lu ", (long unsigned)array_uint64[0]);
    for (i = 1; i < (n + 2 * offset); ++i) printf(", %lu ", (long unsigned)array_uint64[i]);
    printf("];\n");


    return 0;
}
