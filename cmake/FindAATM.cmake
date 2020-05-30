# - Find the AATM atmosphere library
#
# First version:
#   Copyright (c) 2019, Ted Kisner
#
# Usage:
#   find_package(AATM [REQUIRED])
#
# It sets the following variables:
#   AATM_FOUND                  ... true if aatm is found on the system
#   AATM_LIBRARIES              ... full paths to all found aatm libraries
#   AATM_INCLUDE_DIRS           ... aatm include directory paths
#
# The following variables will be checked by the function
#   AATM_ROOT                   ... if set, the libraries are exclusively
#                                   searched under this path.
#   AATM_USE_STATIC_LIBS        ... search for static versions of the libs
#

# Check if we can use PkgConfig
find_package(PkgConfig)

# Determine from PKG
if(PKG_CONFIG_FOUND AND NOT AATM_ROOT)
    pkg_check_modules(PKG_AATM QUIET "aatm")
endif()

# Check whether to search static or dynamic libs

set(AATM_LIB_NAME "aatm")

if(${AATM_USE_STATIC_LIBS})
  set(AATM_LIB_NAME "aatm_static")
endif()

if(AATM_ROOT)
    # find libs
    find_library(
        AATM_LIB
        NAMES ${AATM_LIB_NAME}
        PATHS ${AATM_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
    )

    # find includes
    find_path(AATM_INCLUDE_DIRS
        NAMES "ATMVersion.h"
        PATHS ${AATM_ROOT}
        PATH_SUFFIXES "include"
        NO_DEFAULT_PATH
    )
else(AATM_ROOT)
    find_library(
        AATM_LIB
        NAMES ${AATM_LIB_NAME}
        PATHS "${PKG_AATM_PREFIX}"
        PATH_SUFFIXES "lib" "lib64"
    )

    find_path(AATM_INCLUDE_DIRS
        NAMES "ATMVersion.h"
        PATHS "${PKG_AATM_PREFIX}"
        PATH_SUFFIXES "include"
    )
endif(AATM_ROOT)

if(AATM_LIB)
    set(AATM_LIBRARIES ${AATM_LIB})
endif(AATM_LIB)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(AATM
    REQUIRED_VARS AATM_INCLUDE_DIRS AATM_LIB
    HANDLE_COMPONENTS
)

mark_as_advanced(
    AATM_INCLUDE_DIRS
    AATM_LIBRARIES
)
