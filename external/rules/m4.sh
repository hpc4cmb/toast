curl -SL https://ftp.gnu.org/gnu/m4/m4-1.4.18.tar.bz2 \
    -o m4-1.4.18.tar.bz2 \
    && tar xjf m4-1.4.18.tar.bz2 \
    && cd m4-1.4.18 \
    && CC="gcc" ./configure \
    --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf m4*
