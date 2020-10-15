
// This test main() is used for checking fortran compiler name mangling
// in the LAPACK library without actually using the fortran compiler.

#if defined LAPACK_UPPER
# define func DPOTRF
#elif defined LAPACK_LOWER
# define func dpotrf
#elif defined LAPACK_UBACK
# define func dpotrf_
#elif defined LAPACK_UFRONT
# define func _dpotrf
#endif // if defined LAPACK_UPPER
extern "C" {
void func(char * UPLO, int * N, double * A, int * LDA, int * INFO);
}
int main(int argc, char ** argv) {
    char UPLO = 'L';
    int N = 1;
    double A[1];
    int LDA = 1;
    int INFO = 0;
    func(&UPLO, &N, A, &LDA, &INFO);
    return 0;
}
