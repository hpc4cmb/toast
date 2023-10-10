# Install Openblas

# Variables set by code sourcing this:
# - CC
# - CFLAGS
# - FC
# - FCFLAGS
# - FCLIBS
# - OMPFLAGS
# - PREFIX
# - STATIC (yes/no)
# - MAKEJ
# - CLEANUP (yes/no)

openblas_version=0.3.23
openblas_dir=OpenBLAS-${openblas_version}
openblas_pkg=${openblas_dir}.tar.gz

if [ ! -e ${openblas_pkg} ]; then
    echo "Fetching OpenBLAS..."
    curl -SL https://github.com/xianyi/OpenBLAS/archive/v${openblas_version}.tar.gz -o ${openblas_pkg}
fi

echo "Building OpenBLAS..."

shr="NO_STATIC=1"
targ="libs netlib shared"
if [ "${STATIC}" = "yes" ]; then
    shr="NO_SHARED=1"
    targ="libs netlib"
fi

start_dir=$(pwd)
rm -rf ${openblas_dir}
tar xzf ${openblas_pkg} \
    && pushd ${openblas_dir} >/dev/null 2>&1 \
    && make USE_OPENMP=1 ${shr} \
    MAKE_NB_JOBS=${MAKEJ} \
    CC="${CC}" FC="${FC}" DYNAMIC_ARCH=1 TARGET=GENERIC \
    COMMON_OPT="${CFLAGS}" FCOMMON_OPT="${FCFLAGS}" \
    EXTRALIB="${OMPFLAGS} -lm ${FCLIBS}" ${targ} \
    && make ${shr} DYNAMIC_ARCH=1 TARGET=GENERIC PREFIX="${PREFIX}" install \
    && popd >/dev/null 2>&1
pushd "${start_dir}" >/dev/null 2>&1

if [ "x${CLEANUP}" = "xyes" ]; then
    rm -rf ${openblas_dir}
fi
