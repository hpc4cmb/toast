curl -SL http://www.fftw.org/fftw-3.3.5.tar.gz \
    -o fftw-3.3.5.tar.gz \
    && tar xzf fftw-3.3.5.tar.gz \
    && cd fftw-3.3.5 \
    && CC="@CC@" CFLAGS="@CFLAGS@" ./configure --enable-threads @CROSS@ --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf fftw*
