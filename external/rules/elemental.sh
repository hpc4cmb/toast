curl -SL https://github.com/elemental/Elemental/archive/v0.87.7.tar.gz \
    -o Elemental-0.87.7.tar.gz \
    && tar -xzf Elemental-0.87.7.tar.gz \
    && cd Elemental-0.87.7 \
    && mkdir build && cd build \
    && cmake \
    -D CMAKE_INSTALL_PREFIX="@AUX_PREFIX@" \
    -D INSTALL_PYTHON_PACKAGE=OFF \
    -D CMAKE_CXX_COMPILER="@CXX@" \
    -D CMAKE_C_COMPILER="@CC@" \
    -D CMAKE_Fortran_COMPILER="@FC@" \
    -D MPI_CXX_COMPILER="@MPICXX@" \
    -D MPI_C_COMPILER="@MPICC@" \
    -D MPI_Fortran_COMPILER="@MPIFC@" \
    -D METIS_GKREGEX=ON \
    -D MATH_LIBS="$(echo '@LAPACK@ @BLAS@' | sed -e 's#^ ##g')" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_FLAGS="@CXXFLAGS@" \
    .. \
    && make && make install \
    && cd ../.. \
    && rm -rf Elemental*
