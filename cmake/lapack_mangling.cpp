
// This test main() is used for checking fortran compiler name mangling
// in the LAPACK library without actually using the fortran compiler.

#if defined LAPACK_UPPER
# define func ILAVER
#elif defined LAPACK_LOWER
# define func ilaver
#elif defined LAPACK_UBACK
# define func ilaver_
#elif defined LAPACK_UFRONT
# define func _ilaver
#endif // if defined LAPACK_UP
extern "C" {
void func(int * major, int * minor, int * patch);
}
int main(int argc, char ** argv) {
    int major = 0;
    int minor = 0;
    int patch = 0;
    func(&major, &minor, &patch);
    return 0;
}
