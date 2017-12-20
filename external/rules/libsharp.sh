git clone https://github.com/zonca/libsharp --branch almxfl --single-branch --depth 1 \
    && cd libsharp \
    && patch -p1 < ../rules/patch_libsharp \
    && autoreconf \
    && CC="@MPICC@" CFLAGS="@CFLAGS@" \
    ./configure @CROSS@ --enable-mpi --enable-pic --prefix="@AUX_PREFIX@" \
    && make \
    && cp -a auto/* "@AUX_PREFIX@/" \
    && cd python \
    && LIBSHARP="@AUX_PREFIX@" CC="@MPICC@ -g" LDSHARED="@MPICC@ -g -shared" \
    python setup.py install --prefix="@AUX_PREFIX@" \
    && cd ../.. \
    && rm -rf libsharp*
