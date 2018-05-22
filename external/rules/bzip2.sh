curl -SL http://www.bzip.org/1.0.6/bzip2-1.0.6.tar.gz \
    | tar xzf - \
    && cd bzip2-1.0.6 \
    && patch -p1 < ../rules/patch_bzip2 \
    && CC="@CC@" CFLAGS="@CFLAGS@" \
    make -f Makefile-toast \
    && cp -a bzlib.h "@AUX_PREFIX@/include" \
    && cp -a libbz2.so* "@AUX_PREFIX@/lib" \
    && cd .. \
    && rm -rf bzip2*
