curl -SL https://ftp.gnu.org/gnu/autoconf/autoconf-2.69.tar.gz \
    -o autoconf-2.69.tar.gz \
    && tar xzf autoconf-2.69.tar.gz \
    && cd autoconf-2.69 \
    && CC="gcc" ./configure @CROSS@ \
    --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf autoconf*
