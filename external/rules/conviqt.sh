curl -SL https://www.dropbox.com/s/4tzjn9bgq7enkf9/libconviqt-1.1.0.tar.bz2?dl=0 \
    | tar -xjf - \
    && cd libconviqt-1.1.0 \
    && CC="@MPICC@" CXX="@MPICXX@" MPICC="@MPICC@" MPICXX="@MPICXX@" \
    CFLAGS="@CFLAGS@ -std=gnu99" CXXFLAGS="@CXXFLAGS@" \
    ./configure @CROSS@ --with-cfitsio="@AUX_PREFIX@" --prefix="@AUX_PREFIX@" \
    && make -j @MAKEJ@ && make install \
    $$ cd python
    $$ python setup.py install --prefix=@AUX_PREFIX@ \
    && cd ../.. \
    && rm -rf libconviqt*
