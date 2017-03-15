curl -SL https://github.com/hpc4cmb/libmadam/releases/download/0.2.0/libmadam-0.2.0.tar.bz2 \
    -o libmadam-0.2.0.tar.bz2 \
    && tar -xjf libmadam-0.2.0.tar.bz2 \
    && cd libmadam-0.2.0 \
    && FC="@MPIFC@" MPIFC="@MPIFC@" FCFLAGS="@FCFLAGS@" \
    ./configure @CROSS@ --with-cfitsio="@AUX_PREFIX@" \
    --with-blas="@BLAS@" --with-lapack="@LAPACK@" \
    --with-fftw="@AUX_PREFIX@" --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf libmadam*
