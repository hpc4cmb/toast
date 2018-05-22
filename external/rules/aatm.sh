curl -SL https://launchpad.net/aatm/trunk/0.5/+download/aatm-0.5.tar.gz \
    | tar xzf - \
    && cd aatm-0.5 \
    && chmod -R u+w . \
    && patch -p1 < ../rules/patch_aatm \
    && autoreconf \
    && CC="@CC@" CFLAGS="@CFLAGS@" \
    CPPFLAGS="-I@AUX_PREFIX@/include" \
    LDFLAGS="-L@AUX_PREFIX@/lib" \
    ./configure @CROSS@ \
    --prefix="@AUX_PREFIX@" \
    && make -j @MAKEJ@ && make install \
    && cd .. \
    && rm -rf aatm*
