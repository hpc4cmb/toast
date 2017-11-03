# - Find a LAPACK library (no includes)
# This module defines
#  LAPACK_LIBRARIES, the libraries needed to use LAPACK.
#  LAPACK_FOUND, If false, do not try to use LAPACK.
# also defined, but not for general use are
#  LAPACK_LIBRARY, where to find the LAPACK library.

FIND_LIBRARY(LAPACK_LIBRARY
    NAMES lapack

    HINTS
    ${LAPACK_ROOT}
    $ENV{LAPACK_ROOT}
    
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
find_package_handle_standard_args(LAPACK DEFAULT_MSG LAPACK_LIBRARY)

set(LAPACK_LIBRARIES ${LAPACK_LIBRARY})
MARK_AS_ADVANCED(LAPACK_LIBRARY)

