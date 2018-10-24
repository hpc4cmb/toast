curl -SL https://ftp.gnu.org/gnu/autoconf/autoconf-2.69.tar.gz \
    | tar xzf - \
    && cd autoconf-2.69 \
    && CC="gcc" ./configure \
    --prefix="@AUX_PREFIX@" \
    && make -j @MAKEJ@ && make install \
    && cd .. \
    && rm -rf autoconf*
