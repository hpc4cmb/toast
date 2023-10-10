# Install FFTW

# Variables set by code sourcing this:
# - CC
# - CFLAGS
# - FC
# - FCFLAGS
# - OMPFLAGS
# - PREFIX
# - STATIC (yes/no)
# - MAKEJ
# - CLEANUP (yes/no)

fftw_version=3.3.10
fftw_dir=fftw-${fftw_version}
fftw_pkg=${fftw_dir}.tar.gz

echo "Fetching FFTW..."

if [ ! -e ${fftw_pkg} ]; then
    curl -SL http://www.fftw.org/${fftw_pkg} -o ${fftw_pkg}
fi

echo "Building FFTW..."

shr="--enable-shared --disable-static"
if [ "${STATIC}" = "yes" ]; then
    shr="--enable-static --disable-shared"
fi

threads="--enable-openmp"
if [ "x${OMPFLAGS}" = "x" ]; then
    threads="--enable-threads"
fi

rm -rf ${fftw_dir}
tar xzf ${fftw_pkg} \
    && pushd ${fftw_dir} >/dev/null 2>&1 \
    && CC="${CC}" CFLAGS="${CFLAGS}" \
    FC="${FC}" FCFLAGS="${FCFLAGS}" \
    ./configure \
    --enable-fortran \
    ${threads} ${shr} \
    --prefix="${PREFIX}" \
    && make -j ${MAKEJ} \
    && make install \
    && popd >/dev/null 2>&1

if [ "x${CLEANUP}" = "xyes" ]; then
    rm -rf ${fftw_dir}
fi
