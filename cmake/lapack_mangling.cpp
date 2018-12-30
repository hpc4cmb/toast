
// This test main() is used for checking fortran compiler name mangling
// in the LAPACK library without actually using the fortran compiler.

#if defined LAPACK_UPPER
# define ilaver ILAVER
#elif defined LAPACK_LOWER
# define ilaver ilaver
#elif defined LAPACK_UBACK
# define ilaver ilaver_
#elif defined LAPACK_UFRONT
# define ilaver _ilaver
#endif // if defined LAPACK_UP
extern "C" {
void ilaver(int * major, int * minor, int * patch);
}
int main(int argc, char ** argv) {
    int major = 0;
    int minor = 0;
    int patch = 0;
    ilaver(&major, &minor, &patch);
    return 0;
}
