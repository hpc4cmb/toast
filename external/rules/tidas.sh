git clone https://github.com/hpc4cmb/tidas.git --branch master --single-branch --depth 1 \
    && cd tidas \
    && mkdir build \
    && cd build \
    && LDFLAGS='-dynamic' \
    cmake \
    -DCMAKE_C_COMPILER="@MPICC@" \
    -DCMAKE_CXX_COMPILER="@MPICXX@" \
    -DMPI_C_COMPILER="@MPICC@" \
    -DMPI_CXX_COMPILER="@MPICXX@" \
    -DCMAKE_C_FLAGS="@CFLAGS@ -dynamic -pthread -DSQLITE_DISABLE_INTRINSIC" \
    -DCMAKE_CXX_FLAGS="@CXXFLAGS@ -dynamic -pthread -DSQLITE_DISABLE_INTRINSIC" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DPYTHON_EXECUTABLE:FILEPATH="@CONDA_PREFIX@/bin/python" \
    -DCMAKE_INSTALL_PREFIX="@AUX_PREFIX@" \
    .. \
    && make -j 4 && make install \
    && cd ../.. \
    && rm -rf tidas
