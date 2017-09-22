curl -SL https://support.hdfgroup.org/ftp/HDF5/current18/src/hdf5-1.8.19.tar.bz2 \
    | tar xjf - \
    && cd hdf5-1.8.19 \
    && CC="@CC@" CFLAGS=$(if [ "x@CROSS@" = x ]; then echo "@CFLAGS@"; \
       else echo "-O3"; fi) \
    CXX="@CXX@" CXXFLAGS=$(if [ "x@CROSS@" = x ]; then echo "@CXXFLAGS@"; \
       else echo "-O3"; fi) \
    ./configure --disable-fortran --disable-fortran2003 \
    --disable-silent-rules \
    --disable-parallel \
    --enable-cxx \
    --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf hdf5*
