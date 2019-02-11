curl -SL https://github.com/hpc4cmb/libmadam/releases/download/v1.0.1/libmadam-1.0.1.tar.bz2 \
    | tar -xjf - \
    && cd libmadam-1.0.1 \
    && FC="@MPIFC@" MPIFC="@MPIFC@" FCFLAGS="@FCFLAGS@" \
    CC="@MPICC@" MPICC="@MPICC@" CFLAGS="@CFLAGS@" \
    ./configure @CROSS@ --with-cfitsio="@AUX_PREFIX@" \
    --with-blas="@BLAS@" --with-lapack="@LAPACK@" \
    --with-fftw="@AUX_PREFIX@" --prefix="@AUX_PREFIX@" \
    && make -j @MAKEJ@ && make install \
    && cd python \
    && python setup.py install --prefix "@AUX_PREFIX@" \
    && cd ../.. \
    && rm -rf libmadam*
