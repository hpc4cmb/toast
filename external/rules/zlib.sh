curl -SL http://zlib.net/zlib-1.2.11.tar.gz \
    -o zlib-1.2.11.tar.gz \
    && tar xzf zlib-1.2.11.tar.gz \
    && cd zlib-1.2.11 \
    && CC="@CC@" CFLAGS="@CFLAGS@" \
    ./configure --shared --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf zlib*
