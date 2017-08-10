curl -SL https://github.com/hpc4cmb/libmadam/releases/download/0.2.3/libmadam-0.2.3.tar.bz2 \
    | tar -xjf - \
    && cd libmadam-0.2.3 \
    && FC="@MPIFC@" MPIFC="@MPIFC@" FCFLAGS="@FCFLAGS@" \
    CC="@MPICC@" MPICC="@MPICC@" CFLAGS="@CFLAGS@" \
    ./configure @CROSS@ --with-cfitsio="@AUX_PREFIX@" \
    --with-blas="@BLAS@" --with-lapack="@LAPACK@" \
    --with-fftw="@AUX_PREFIX@" --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf libmadam*
