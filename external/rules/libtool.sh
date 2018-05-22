curl -SL https://ftp.gnu.org/gnu/libtool/libtool-2.4.6.tar.gz \
    | tar xzf - \
    && cd libtool-2.4.6 \
    && CC="gcc" ./configure \
    --prefix="@AUX_PREFIX@" \
    && make -j @MAKEJ@ && make install \
    && cd .. \
    && rm -rf libtool*
