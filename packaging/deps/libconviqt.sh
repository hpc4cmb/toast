# Install libconviqt

# Variables set by code sourcing this:
# - MPICC
# - CFLAGS
# - MPICXX
# - CXXFLAGS
# - PREFIX
# - STATIC (yes/no)
# - MAKEJ
# - CLEANUP (yes/no)

conviqt_version=1.2.7
conviqt_dir="libconviqt-${conviqt_version}"
conviqt_pkg="${conviqt_dir}.tar.gz"

if [ ! -e ${conviqt_pkg} ]; then
    curl -SL -o ${conviqt_pkg} https://github.com/hpc4cmb/libconviqt/archive/v${conviqt_version}.tar.gz
fi

rm -rf ${conviqt_dir}
tar xzf ${conviqt_pkg} \
    && pushd ${conviqt_dir} >/dev/null \
    && ./autogen.sh \
    && CC="${MPICC}" CXX="${MPICXX}" MPICC="${MPICC}" MPICXX="${MPICXX}" \
    CFLAGS="${CFLAGS} -std=gnu99" CXXFLAGS="${CXXFLAGS}" \
    LDFLAGS="-L${PREFIX}/lib" \
    ./configure \
    --with-cfitsio="${PREFIX}" --prefix="${PREFIX}" \
    && make -j ${MAKEJ} \
    && make install \
    && pushd python >/dev/null \
    && python3 setup.py install --prefix="${PREFIX}" --old-and-unmanageable \
    && popd >/dev/null \
    && popd >/dev/null

if [ "x${CLEANUP}" = "xyes" ]; then
    rm -rf ${conviqt_dir}
fi
