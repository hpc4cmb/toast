curl -SL http://faculty.cse.tamu.edu/davis/SuiteSparse/SuiteSparse-5.4.0.tar.gz \
    -o SuiteSparse-5.4.0.tar.gz \
    && tar xzf SuiteSparse-5.4.0.tar.gz \
    && cd SuiteSparse \
    && make CC="@CC@" CXX="@CXX@" CFLAGS="@CFLAGS@" AUTOCC=no \
    F77="@FC@" F77FLAGS="@FCFLAGS" \
    CFOPENMP="@OPENMP_CXXFLAGS@" LAPACK="@LAPACK@" BLAS="@BLAS@" \
    && make install CC="@CC@" CXX="@CXX@" CFLAGS="@CFLAGS@" AUTOCC=no \
    F77="@FC@" F77FLAGS="@FCFLAGS" \
    CFOPENMP="@OPENMP_CXXFLAGS@" LAPACK="@LAPACK@" BLAS="@BLAS@" \
    INSTALL="@AUX_PREFIX@" \
    && cd .. \
    && rm -rf SuiteSparse*
