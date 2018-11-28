curl -SL https://github.com/xianyi/OpenBLAS/archive/v0.3.3.tar.gz \
    | tar xzf - \
    && cd OpenBLAS-0.3.3 \
    && make USE_OPENMP=1 NO_SHARED=1 \
    FC="@FC@" FCFLAGS="@FCFLAGS@" \
    CC="@CC@" CFLAGS="@CFLAGS@" \
    && make NO_SHARED=1 PREFIX="@AUX_PREFIX@" install \
    && cd .. \
    && rm -rf OpenBLAS*
