curl -SL https://pypi.python.org/packages/ee/b8/f443e1de0b6495479fc73c5863b7b5272a4ece5122e3589db6cd3bb57eeb/mpi4py-2.0.0.tar.gz#md5=4f7d8126d7367c239fd67615680990e3 \
    | tar xzf - \
    && cd mpi4py-2.0.0 \
    && echo $'\n\
[toast]\n\
mpicc = @MPICC@\n\
mpicxx = @MPICXX@\n\
include_dirs = @MPI_CPPFLAGS@\n\
library_dirs = @MPI_LDFLAGS@\n\
runtime_library_dirs = @MPI_LDFLAGS@\n\
libraries = @MPI_LIB@\n\
extra_compile_args = @MPI_EXTRA_COMP@\n\
extra_link_args = @MPI_EXTRA_LINK@\n\
' > mpi.cfg \
    && python setup.py build --mpi=toast \
    && python setup.py install --prefix=@CONDA_PREFIX@ \
    && cd .. \
    && rm -rf mpi4py*
