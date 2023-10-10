# Install CFITSIO

# Variables set by code sourcing this:
# - CC
# - CFLAGS
# - PREFIX
# - STATIC (yes/no)
# - CLEANUP (yes/no)

cfitsio_version=4.3.0
cfitsio_dir=cfitsio-${cfitsio_version}
cfitsio_pkg=${cfitsio_dir}.tar.gz

echo "Fetching CFITSIO..."

if [ ! -e ${cfitsio_pkg} ]; then
    curl -SL https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/${cfitsio_pkg} -o ${cfitsio_pkg}
fi

echo "Building CFITSIO..."

rm -rf ${cfitsio_dir}
tar xzf ${cfitsio_pkg} \
    && pushd ${cfitsio_dir} >/dev/null 2>&1 \
    && CC="${CC}" CFLAGS="${CFLAGS}" \
    ./configure \
    --enable-reentrant \
    --prefix="${PREFIX}" \
    && make stand_alone \
    && make utils \
    && if [ "x${STATIC}" != "xyes" ]; then make shared; fi \
    && make install \
    && popd >/dev/null 2>&1

if [ "x${CLEANUP}" = "xyes" ]; then
    rm -rf ${cfitsio_dir}
fi
