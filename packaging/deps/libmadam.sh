# Install libmadam

# Variables set by code sourcing this:
# - MPICC
# - CFLAGS
# - MPIFC
# - FCFLAGS
# - MPFCLIBS
# - PREFIX
# - STATIC (yes/no)
# - MAKEJ
# - CLEANUP (yes/no)

# LDFLAGS="-L${PREFIX}/lib -lfmpich -lgfortran" \

madam_version=1.0.2
madam_dir="libmadam-${madam_version}"
madam_pkg=${madam_dir}.tar.bz2

if [ ! -e ${madam_pkg} ]; then
    curl -SL -o ${madam_pkg} https://github.com/hpc4cmb/libmadam/releases/download/v${madam_version}/${madam_pkg}
fi

rm -rf ${madam_dir}
tar xjf ${madam_pkg} \
    && pushd ${madam_dir} >/dev/null \
    && FC="${MPIFC}" MPIFC="${MPIFC}" FCFLAGS="${FCFLAGS}" \
    CC="${MPICC}" MPICC="${MPICC}" CFLAGS="${CFLAGS}" \
    LDFLAGS="${MPFCLIBS}" \
    ./configure --with-cfitsio="${PREFIX}" \
    --with-fftw="${PREFIX}" \
    --with-blas="-L${PREFIX}/lib -lopenblas" \
    --with-lapack="-L${PREFIX}/lib -lopenblas" \
    --prefix="${PREFIX}" \
    && make -j ${MAKEJ} \
    && make install \
    && pushd python >/dev/null \
    && python3 setup.py install --prefix="${PREFIX}" --old-and-unmanageable \
    && popd >/dev/null \
    && popd >/dev/null

if [ "x${CLEANUP}" = "xyes" ]; then
    rm -rf ${madam_dir}
fi
