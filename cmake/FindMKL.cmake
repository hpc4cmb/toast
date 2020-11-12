# - Find INTEL MKL Single Dynamic Library
#
# First version:
#   Copyright (c) 2020, Theodore Kisner
#
# Usage:
#   find_package(MKL [REQUIRED])
#
# It sets the following variables:
#   MKL_FOUND                  ... true if MKL is found on the system
#   MKL_LIBRARIES              ... full paths to all found MKL libraries
#   MKL_INCLUDE_DIRS           ... MKL include directory paths
#
# The following variables will be checked by the function
#   MKL_DISABLED               ... if true, skip all checks for MKL.
#   MKL_DIR                    ... if set, look for MKL in this directory.
#   environment $MKLROOT       ... if set, look for MKL in this directory.
#

set(MKL_FOUND FALSE)
if(MKL_DISABLED)
    message("-- MKL disabled, skipping checks")
else(MKL_DISABLED)

# Set default value of MKL_DIR.

if(NOT MKL_DIR)
    if(WIN32)
        if(DEFINED ENV{MKLProductDir})
            set(MKL_DIR "$ENV{MKLProductDir}/mkl")
        else()
            set(
                MKL_DIR
                "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl"
            )
        endif()
    else()
        set(MKL_DIR "/opt/intel/mkl")
        if(DEFINED ENV{MKLROOT})
            # This is the standard MKL environment variable.
            set(MKL_DIR "$ENV{MKLROOT}")
        endif()
    endif()
endif()

find_package(Threads)

# First look in user-specified or default location

if(NOT MKL_FOUND)
    set(MKL_LIB_DIR "${MKL_DIR}/lib")
    find_library(
        MKL_RT_LIB
        NAMES "mkl_rt"
        PATHS "${MKL_LIB_DIR}"
        PATH_SUFFIXES "intel64"
        NO_DEFAULT_PATH
    )
    # find includes
    find_path(MKL_INCLUDE_DIRS
        NAMES "mkl_dfti.h"
        PATHS ${MKL_DIR}
        PATH_SUFFIXES "include"
        NO_DEFAULT_PATH
    )
    if(MKL_RT_LIB)
        set(MKL_FOUND TRUE)
    endif()
endif()

# If we did not find it, search everywhere

if(NOT MKL_FOUND)
    find_library(
        MKL_RT_LIB
        NAMES "mkl_rt"
    )
    # find includes
    find_path(MKL_INCLUDE_DIRS
        NAMES "mkl_dfti.h"
    )
    if(MKL_RT_LIB)
        set(MKL_FOUND TRUE)
    endif()
endif()

if(MKL_FOUND)
    set(MKL_LIBRARIES
	${MKL_RT_LIB} ${CMAKE_THREAD_LIBS_INIT} m ${CMAKE_DL_LIBS}
	CACHE STRING "MKL dynamic libraries"
    )
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(MKL
    REQUIRED_VARS MKL_INCLUDE_DIRS MKL_LIBRARIES
    HANDLE_COMPONENTS
)

mark_as_advanced(
    MKL_INCLUDE_DIRS
    MKL_LIBRARIES
)

endif(MKL_DISABLED)

