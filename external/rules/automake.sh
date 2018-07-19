curl -SL http://ftp.gnu.org/gnu/automake/automake-1.16.1.tar.gz \
    | tar xzf - \
    && cd automake-1.16.1 \
    && CC="gcc" ./configure \
    --prefix="@AUX_PREFIX@" \
    && make -j @MAKEJ@ && make install \
    && cd .. \
    && rm -rf automake*
