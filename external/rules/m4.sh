curl -SL https://ftp.gnu.org/gnu/m4/m4-1.4.18.tar.bz2 \
    | tar xjf - \
    && cd m4-1.4.18 \
    && CC="gcc" ./configure \
    --prefix="@AUX_PREFIX@" \
    && make -j @MAKEJ@ && make install \
    && cd .. \
    && rm -rf m4*
