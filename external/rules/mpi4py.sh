curl -SL https://pypi.python.org/packages/ee/b8/f443e1de0b6495479fc73c5863b7b5272a4ece5122e3589db6cd3bb57eeb/mpi4py-2.0.0.tar.gz#md5=4f7d8126d7367c239fd67615680990e3 \
    -o mpi4py-2.0.0.tar.gz \
    && tar xzf mpi4py-2.0.0.tar.gz \
    && cd mpi4py-2.0.0 \
    && echo $'
[toast]
mpicc = @MPICC@
mpicxx = @MPICXX@
include_dirs = @MPI_CPPFLAGS@
library_dirs = @MPI_LDFLAGS@
runtime_library_dirs = @MPI_LDFLAGS@
libraries = @MPI_LIB@
extra_compile_args = @MPI_EXTRA_COMP@
extra_link_args = @MPI_EXTRA_LINK@
' > mpi.cfg \
    && python setup.py build --mpi=toast \
    && python setup.py install --prefix=@CONDA_PREFIX@ \
    && cd .. \
    && rm -rf mpi4py*
