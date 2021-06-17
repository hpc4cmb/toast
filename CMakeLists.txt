# TOAST

# This minimum version is mostly set in order to get a newer version
# of the FindMPI check.  Note that you can easily install a newer cmake version
# using conda or pip.
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

foreach(policy
    CMP0048
    CMP0074
    CMP0077
    CMP0063
  )
  if(POLICY ${policy})
    cmake_policy(SET ${policy} NEW)
  endif()
endforeach()

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/src/toast/RELEASE REL_VERSION)
string(REGEX REPLACE "^([0-9]+\\.[0-9]+)\\..*" "\\1" MAJMIN_VERSION "${REL_VERSION}")

project(toast VERSION ${MAJMIN_VERSION} LANGUAGES C CXX)

# Force C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set symbol visibility to hidden to be consistent with pybind11
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Auxiliary files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Use GNUInstallDirs to install libraries into correct locations on all
# platforms.
include(GNUInstallDirs)

# Build defaults
include(BuildType)

# We are building libraries that will eventually be linked into shared
# modules.  All code should be built with PIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# External packages

# In some situations (like building python wheels), it is useful to statically link to
# our external dependencies.  This allows us to ship self-contained compiled
# extensions.  We check a variable here, and if set, we look for static versions of
# our dependencies.
#
if(NOT TOAST_STATIC_DEPS AND NOT $ENV{TOAST_STATIC_DEPS} STREQUAL "")
  set(TOAST_STATIC_DEPS $ENV{TOAST_STATIC_DEPS})
endif()

# OpenMP
find_package(OpenMP)

if(TOAST_STATIC_DEPS)
    set(BLA_STATIC TRUE)
    set(FFTW_USE_STATIC_LIBS TRUE)
    set(AATM_USE_STATIC_LIBS TRUE)
    set(SUITESPARSE_USE_STATIC_LIBS TRUE)
endif()

# First look for MKL, since that will provide both FFT support and BLAS/LAPACK
find_package(MKL)

if(MKL_FOUND)
    # Use MKL for BLAS / LAPACK
    set(BLAS_LIBRARIES "${MKL_LIBRARIES}")
    set(LAPACK_LIBRARIES "${MKL_LIBRARIES}")
    set(BLAS_FOUND TRUE)
    set(LAPACK_FOUND TRUE)
else()
    # Search for FFTW instead
    find_package(FFTW)
    if(NOT FFTW_FOUND)
        message(FATAL_ERROR "Could not find a supported FFT library (MKL or FFTW)")
    endif()
    find_package(BLAS)
    find_package(LAPACK)
endif()

if(BLAS_FOUND)
    if(LAPACK_FOUND)
        find_package(LAPACKnames)
    else()
        if($ENV{READTHEDOCS} STREQUAL "")
            message(FATAL_ERROR "Could not find a working LAPACK installation")
        endif()
    endif()
else()
    if($ENV{READTHEDOCS} STREQUAL "")
        message(FATAL_ERROR "Could not find a working BLAS installation")
    endif()
endif()

find_package(AATM)

find_package(SuiteSparse)

find_package(PythonInterp REQUIRED)

find_package(CUDAToolkit)

# Internal products

enable_testing()
add_subdirectory(src)
add_subdirectory(workflows)
