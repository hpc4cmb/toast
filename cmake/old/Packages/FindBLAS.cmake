# - Find a BLAS library (no includes)
# This module defines
#  BLAS_LIBRARIES, the libraries needed to use BLAS.
#  BLAS_FOUND, If false, do not try to use BLAS.
# also defined, but not for general use are
#  BLAS_LIBRARY, where to find the BLAS library.

FIND_LIBRARY(BLAS_LIBRARY
    NAMES blas

    HINTS
    ${BLAS_ROOT}
    $ENV{BLAS_ROOT}
    /usr

    PATH_SUFFIXES
    ${CMAKE_INSTALL_LIBDIR_DEFAULT}/atlas
    lib64/atlas
    lib/atlas
    ${CMAKE_INSTALL_LIBDIR_DEFAULT}
    lib64
    lib
    
    PATHS 
    /usr/lib64/atlas
    /usr/lib/atlas
    /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLAS DEFAULT_MSG BLAS_LIBRARY)

set(BLAS_LIBRARIES ${BLAS_LIBRARY})
MARK_AS_ADVANCED(BLAS_LIBRARY)

