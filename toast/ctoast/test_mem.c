/* 
This is a simple test executable for low-level testing of C
functions.  Full unit testing is done in python.
*/

#include <stdio.h>
#include <stdlib.h>

#include <pytoast.h>


int main(int argc, char *argv[]) {
    size_t i, j;
    size_t n = 1000;
    double * f64_buf;

    f64_buf = pytoast_mem_aligned_f64(n);

    for (i = 0; i < n; ++i) {
        f64_buf[i] = (double)i;
    }

    pytoast_mem_aligned_free(f64_buf);

    return 0;
}
