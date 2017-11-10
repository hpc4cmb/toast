curl -SL http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio3420.tar.gz \
    | tar xzf - \
    && cd cfitsio \
    && CC="@CC@" CFLAGS="@CFLAGS@" ./configure @CROSS@ \
    --prefix="@AUX_PREFIX@" --enable-reentrant \
    && make stand_alone \
    && make utils \
    && make shared \
    && make install \
    && cd .. \
    && rm -rf cfitsio*
