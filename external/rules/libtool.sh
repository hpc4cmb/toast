curl -SL https://ftp.gnu.org/gnu/libtool/libtool-2.4.6.tar.gz \
    -o libtool-2.4.6.tar.gz \
    && tar xzf libtool-2.4.6.tar.gz \
    && cd libtool-2.4.6 \
    && CC="gcc" ./configure \
    --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf libtool*
