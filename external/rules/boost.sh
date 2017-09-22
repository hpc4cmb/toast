curl -SL https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2 \
    -o boost_1_64_0.tar.bz2 \
    && tar xjf boost_1_64_0.tar.bz2 \
    && cd boost_1_64_0 \
    && echo "" > tools/build/user-config.jam \
    && echo 'using @BOOSTCHAIN@ : : @CXX@ : <cflags>"@CFLAGS@" <cxxflags>"@CXXFLAGS@" ;' >> tools/build/user-config.jam \
    && echo 'using mpi : @MPICXX@ : <include>"@MPI_CPPFLAGS@" <library-path>"@MPI_LDFLAGS@" <find-shared-library>"@MPI_CXXLIB@" <find-shared-library>"@MPI_LIB@" ;' >> tools/build/user-config.jam \
    && BOOST_BUILD_USER_CONFIG=tools/build/user-config.jam ./bootstrap.sh \
    --with-toolset=@BOOSTCHAIN@ \
    --with-python=python@PYVERSION@ \
    --prefix=@AUX_PREFIX@ \
    && ./b2 --layout=tagged \
    $(python@PYVERSION@-config --includes | sed -e 's/-I//g' -e 's/\([^[:space:]]\+\)/ include=\1/g') \
    variant=release threading=multi link=shared runtime-link=shared install \
    && cd .. \
    && rm -rf boost*
