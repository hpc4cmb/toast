curl -SL ftp://ftp.atnf.csiro.au/pub/software/wcslib/wcslib-5.16.tar.bz2 \
    | tar xjf - \
    && cd wcslib-5.16 \
    && chmod -R u+w . \
    && patch -p1 < ../rules/patch_wcslib \
    && autoconf \
    && CC="@CC@" CFLAGS="@CFLAGS@" \
    CPPFLAGS="-I@AUX_PREFIX@/include" \
    LDFLAGS="-L@AUX_PREFIX@/lib" \
    ./configure @CROSS@ \
    --disable-fortran \
    --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf wcslib*
