curl -SL https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.20/src/hdf5-1.8.20.tar.bz2 \
    | tar xjf - \
    && cd hdf5-1.8.20 \
    && CC="@CC@" CFLAGS=$(if [ "x@CROSS@" = x ]; then echo "@CFLAGS@"; \
       else echo "-O3"; fi) \
    CXX="@CXX@" CXXFLAGS=$(if [ "x@CROSS@" = x ]; then echo "@CXXFLAGS@"; \
       else echo "-O3"; fi) \
    FC="@FC@" CXXFLAGS=$(if [ "x@CROSS@" = x ]; then echo "@FCFLAGS@"; \
       else echo "-O3"; fi) \
    ./configure \
    --disable-silent-rules \
    --disable-parallel \
    --enable-cxx \
    --enable-fortran \
    --enable-fortran2003 \
    --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf hdf5*
