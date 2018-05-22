git clone https://github.com/hpc4cmb/toast.git \
    && cd toast \
    && ./autogen.sh \
    && CC=mpicc CXX=mpicxx MPICC=mpicc MPICXX=mpicxx \
    CFLAGS="@CFLAGS@" \
    CXXFLAGS="@CXXFLAGS@" \
    OPENMP_CFLAGS="@OPENMP_CFLAGS@" \
    OPENMP_CXXFLAGS="@OPENMP_CXXFLAGS@" \
    LDFLAGS="-lpthread" \
    ./configure \
    --with-elemental=/usr \
    --with-tbb=no \
    --prefix=/usr \
    && make -j @MAKEJ@ && make install \
    && cd ..
