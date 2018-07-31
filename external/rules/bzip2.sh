curl -SL https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/bzip2/1.0.6-8/bzip2_1.0.6.orig.tar.bz2 \
    | tar xjf - \
    && cd bzip2-1.0.6 \
    && patch -p1 < ../rules/patch_bzip2 \
    && CC="@CC@" CFLAGS="@CFLAGS@" \
    make -f Makefile-toast \
    && cp -a bzlib.h "@AUX_PREFIX@/include" \
    && cp -a libbz2.so* "@AUX_PREFIX@/lib" \
    && cd .. \
    && rm -rf bzip2*
