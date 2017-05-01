curl -SL https://ftp.gnu.org/gnu/automake/automake-1.15.tar.gz \
    -o automake-1.15.tar.gz \
    && tar xzf automake-1.15.tar.gz \
    && cd automake-1.15 \
    && CC="gcc" ./configure \
    --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf automake*
