# Install SuiteSparse - only the pieces we need.

# Variables set by code sourcing this:
# - CC
# - CFLAGS
# - CXX
# - CXXFLAGS
# - FC
# - FCFLAGS
# - FCLIBS
# - OMPFLAGS
# - PREFIX
# - DEPSDIR (to find patches)
# - MAKEJ
# - STATIC (yes/no)
# - SHLIBEXT
# - CLEANUP (yes/no)

ssparse_version=7.1.0
ssparse_dir=SuiteSparse-${ssparse_version}
ssparse_pkg=${ssparse_dir}.tar.gz

echo "Fetching SuiteSparse..."

if [ ! -e ${ssparse_pkg} ]; then
    curl -SL https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v${ssparse_version}.tar.gz -o ${ssparse_pkg}
fi

echo "Building SuiteSparse..."

shr="-DNSTATIC:BOOL=ON -DBLA_STATIC:BOOL=OFF"
if [ "${STATIC}" = "yes" ]; then
    shr="-DNSTATIC:BOOL=OFF -DBLA_STATIC:BOOL=ON"
fi

cmake_opts=" \
    -DCMAKE_C_COMPILER=\"${CC}\" \
    -DCMAKE_CXX_COMPILER=\"${CXX}\" \
    -DCMAKE_Fortran_COMPILER=\"${FC}\" \
    -DCMAKE_C_FLAGS=\"${CFLAGS}\" \
    -DCMAKE_CXX_FLAGS=\"${CXXFLAGS}\" \
    -DCMAKE_Fortran_FLAGS=\"${FCFLAGS}\" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCMAKE_INSTALL_PATH=\"${PREFIX}\" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBLA_VENDOR=OpenBLAS ${shr} \
    -DBLAS_LIBRARIES=\"-L${PREFIX}/lib -lopenblas ${OMPFLAGS} -lm ${FCLIBS}\" \
    -DLAPACK_LIBRARIES=\"-L${PREFIX}/lib -lopenblas ${OMPFLAGS} -lm ${FCLIBS}\" \
    "

rm -rf ${ssparse_dir}
tar xzf ${ssparse_pkg} \
    && pushd ${ssparse_dir} >/dev/null 2>&1 \
    && patch -p1 < "${DEPSDIR}/suitesparse.patch" \
    && for pkg in SuiteSparse_config AMD CAMD CCOLAMD COLAMD CHOLMOD; do \
    pushd ${pkg} >/dev/null 2>&1; \
    CC="${CC}" CX="${CXX}" JOBS=${MAKEJ} \
    CMAKE_OPTIONS=${cmake_opts} \
    make local; \
    make install; \
    popd >/dev/null 2>&1; \
    done \
    && if [ "${STATIC}" = "yes" ]; then cp ./lib/*.a "${PREFIX}/lib/"; \
    else cp ./lib/*.${SHLIBEXT} "${PREFIX}/lib/"; fi \
    && cp ./include/* "${PREFIX}/include/" \
    && popd >/dev/null 2>&1

if [ "x${CLEANUP}" = "xyes" ]; then
    rm -rf ${ssparse_dir}
fi
