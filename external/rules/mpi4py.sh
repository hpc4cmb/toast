curl -SL https://pypi.python.org/packages/31/27/1288918ac230cc9abc0da17d84d66f3db477757d90b3d6b070d709391a15/mpi4py-3.0.0.tar.gz#md5=bfe19f20cef5e92f6e49e50fb627ee70 \
    | tar xzf - \
    && cd mpi4py-3.0.0 \
    && echo "[toast]" > mpi.cfg \
	&& echo "mpicc = @MPICC@" >> mpi.cfg \
	&& echo "mpicxx = @MPICXX@" >> mpi.cfg \
	&& echo "include_dirs = @MPI_CPPFLAGS@" >> mpi.cfg \
	&& echo "library_dirs = @MPI_LDFLAGS@" >> mpi.cfg \
	&& echo "runtime_library_dirs = @MPI_LDFLAGS@" >> mpi.cfg \
	&& echo "libraries = @MPI_LIB@" >> mpi.cfg \
	&& echo "extra_compile_args = @MPI_EXTRA_COMP@" >> mpi.cfg \
	&& echo "extra_link_args = @MPI_EXTRA_LINK@" >> mpi.cfg \
    && python setup.py build --mpi=toast \
    && python setup.py install --prefix=@CONDA_PREFIX@ \
    && cd .. \
    && rm -rf mpi4py*
