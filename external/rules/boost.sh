curl -SL https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.bz2 \
    -o boost_1_65_1.tar.bz2 \
    && tar xjf boost_1_65_1.tar.bz2 \
    && cd boost_1_65_1 \
    && echo "" > tools/build/user-config.jam \
    && echo "using mpi : @MPICXX@ : <include>\"@MPI_CPPFLAGS@\" <library-path>\"@MPI_LDFLAGS@\" <find-shared-library>\"@MPI_CXXLIB@\" <find-shared-library>\"@MPI_LIB@\" ;" >> tools/build/user-config.jam \
    && echo "option jobs : @MAKEJ@ ;" >> tools/build/user-config.jam \
    && BOOST_BUILD_USER_CONFIG=tools/build/user-config.jam \
    BZIP2_INCLUDE="@AUX_PREFIX@/include" \
    BZIP2_LIBPATH="@AUX_PREFIX@/lib" \
    ./bootstrap.sh \
    --with-toolset=@BOOSTCHAIN@ \
    --with-python=python@PYVERSION@ \
    --prefix=@AUX_PREFIX@ \
    && ./b2 --layout=tagged --user-config=./tools/build/user-config.jam\
    $(python3-config --includes | sed -e 's/-I//g' -e 's/\([^[:space:]]\+\)/ include=\1/g') \
    variant=release threading=multi link=shared runtime-link=shared install \
    && cd .. \
    && rm -rf boost*
