curl -SL http://zlib.net/zlib-1.2.11.tar.gz \
    | tar xzf - \
    && cd zlib-1.2.11 \
    && CC="gcc" ./configure --shared \
    --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf zlib*
