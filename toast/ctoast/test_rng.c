#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "pytoast.h"

int main(int argc, char const *argv[])
{
    int i;
    int n = 11, offset = 2;
    uint64_t counter1 = 123, counter2 = 321;
    if(argc>1) counter2 = atoi(argv[1]);
    if(argc>2) counter1 = atoi(argv[2]);
    double * array = malloc( (n + 2 * offset) * sizeof(double) );
    
    generate_grv(n, offset, counter1, counter2, array);

    printf("xg = [ %f ", array[0]);
    for (i = 1; i < (n + 2 * offset); ++i) printf(", %f ", array[i]);
    printf("];\n");


    generate_neg11rv(n, offset, counter1, counter2, array);

    printf("xneg11 = [ %f ", array[0]);
    for (i = 1; i < (n + 2 * offset); ++i) printf(", %f ", array[i]);
    printf("];\n");


    generate_01rv(n, offset, counter1, counter2, array);

    printf("x01 = [ %f ", array[0]);
    for (i = 1; i < (n + 2 * offset); ++i) printf(", %f ", array[i]);
    printf("];\n");


    return 0;
}
