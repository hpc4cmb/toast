git clone https://github.com/hpc4cmb/tidas.git \
    && cd tidas \
    && CC="@CC@" CFLAGS="@CFLAGS@" \
    CXX="@CXX@" CXXFLAGS="@CXXFLAGS@" \
    FC="@FC@" FCFLAGS="@FCFLAGS@" \
    MPICC="@MPICC@" MPICXX="@MPICXX@" MPIFC="@MPIFC@" \
    PYTHON="@CONDA_PREFIX@/bin/python" \
    ./configure @CROSS@ \
    --with-hdf5="@AUX_PREFIX@/bin/h5cc" \
    --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf tidas
