curl -SL https://github.com/elemental/Elemental/archive/v0.87.7.tar.gz \
    | tar -xzf - \
    && cd Elemental-0.87.7 \
    && mkdir build && cd build \
    && cmake \
    -D CMAKE_INSTALL_PREFIX="@AUX_PREFIX@" \
    -D INSTALL_PYTHON_PACKAGE=OFF \
    -D CMAKE_CXX_COMPILER="@MPICXX@" \
    -D CMAKE_C_COMPILER="@MPICC@" \
    -D CMAKE_Fortran_COMPILER="@MPIFC@" \
    -D MPI_CXX_COMPILER="@MPICXX@" \
    -D MPI_C_COMPILER="@MPICC@" \
    -D MPI_Fortran_COMPILER="@MPIFC@" \
    -D METIS_GKREGEX=ON \
    -D EL_DISABLE_PARMETIS=TRUE \
    -D MATH_LIBS="$(echo '@LAPACK@ @BLAS@' | sed -e 's#^ ##g')" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_FLAGS="@CXXFLAGS@ @OPENMP_CXXFLAGS@" \
    -D CMAKE_C_FLAGS="@CFLAGS@ @OPENMP_CFLAGS@" \
    -D CMAKE_Fortran_FLAGS="@FCFLAGS@ @OPENMP_CFLAGS@" \
    -D CMAKE_SHARED_LINKER_FLAGS="@OPENMP_CXXFLAGS@" \
    .. \
    && make -j @MAKEJ@ && make install \
    && cd ../.. \
    && rm -rf Elemental*
