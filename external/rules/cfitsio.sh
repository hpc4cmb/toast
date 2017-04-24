curl -SL http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio3410.tar.gz \
    -o cfitsio3410.tar.gz \
    && tar xzf cfitsio3410.tar.gz \
    && cd cfitsio \
    && CC="@CC@" CFLAGS="@CFLAGS@" ./configure @CROSS@ --prefix="@AUX_PREFIX@" \
    && make -j 4 && make shared && make install \
    && cd .. \
    && rm -rf cfitsio*
