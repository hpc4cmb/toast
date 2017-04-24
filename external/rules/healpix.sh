curl -SL https://github.com/tskisner/healpix-autotools/releases/download/v3.31.1/healpix-autotools-3.31.1.tar.gz \
    -o healpix-autotools-3.31.1.tar.gz \
    && tar -xzf healpix-autotools-3.31.1.tar.gz \
    && cd healpix-autotools-3.31.1 \
    && CC="@CC@" CXX="@CXX@" FC="@FC@" \
    CFLAGS="@CFLAGS@ -std=gnu99" CXXFLAGS="@CXXFLAGS@ -std=c++98" FCFLAGS="@FCFLAGS@" \
    ./configure @CROSS@ --with-cfitsio="@AUX_PREFIX@" --prefix="@AUX_PREFIX@" \
    && make && make install \
    && cd .. \
    && rm -rf healpix*
