curl -SL http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz \
    | tar -xzf - \
    && cd mpich-3.2 \
    && CC="@CC@" CXX="@CXX@" FC="@FC@" \
    CFLAGS="@CFLAGS@" CXXFLAGS="@CXXFLAGS@" FCFLAGS="@FCFLAGS@" \
    ./configure @CROSS@ --prefix=@AUX_PREFIX@ \
    && make -j @MAKEJ@ && make install \
    && cd .. \
    && rm -rf mpich-3.2*
