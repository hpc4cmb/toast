# TOAST

# This minimum version is mostly set in order to get a newer version
# of the FindMPI check.  Note that you can easily install a newer cmake version
# using conda or pip.
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

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

if(OpenMP_CXX_FOUND)
    message(STATUS
        "Found OpenMP version: \"" ${OpenMP_CXX_VERSION}
        "\" spec date " ${OpenMP_CXX_SPEC_DATE}
    )
    # Allow the user to force enabling of target offload.  Some compilers support all
    # the features we need in the 5.0 spec, but return a supported version less than
    # that.
    if(NOT USE_OPENMP_TARGET)
        # Check the version reported by the compiler
        if(OpenMP_CXX_VERSION_MAJOR GREATER_EQUAL 5)
            set(USE_OPENMP_TARGET TRUE)
        else()
            if(OpenMP_CXX_SPEC_DATE GREATER_EQUAL 201811)
                # This is the spec date for the 5.0 standard
                set(USE_OPENMP_TARGET TRUE)
            else()
                set(USE_OPENMP_TARGET FALSE)
            endif()
        endif()
    endif()
    if(USE_OPENMP_TARGET)
        message(STATUS "Enabling support for OpenMP target offload")
    else()
        message(STATUS
            "OpenMP target offload disabled.  Force on with -DUSE_OPENMP_TARGET=TRUE"
        )
    endif()
endif()

if(TOAST_STATIC_DEPS)
    set(BLA_STATIC TRUE)
    set(FFTW_USE_STATIC_LIBS TRUE)
    set(AATM_USE_STATIC_LIBS TRUE)
    set(SUITESPARSE_USE_STATIC_LIBS TRUE)
endif()

# First look for MKL and CUDA, since those will provide both
# FFT support and BLAS/LAPACK.

# CUDA.  In some cases, the CUDA toolkit is available (and will be detected) even
# if we want to build without it.  Force the user to request using CUDA.
if(USE_CUDA)
    find_package(CUDAToolkit)
endif()

if(USE_MKL)
    find_package(MKL)
endif()

if(MKL_FOUND)
    # Use MKL for BLAS / LAPACK
    set(BLAS_LIBRARIES "${MKL_LIBRARIES}")
    set(LAPACK_LIBRARIES "${MKL_LIBRARIES}")
    set(BLAS_FOUND TRUE)
    set(LAPACK_FOUND TRUE)
else()
    if(CUDAToolkit_FOUND)
        # This provides FFT support
	message(STATUS "Using CUDA FFT")
    else()
        # Search for FFTW instead
        find_package(FFTW)
        if(NOT FFTW_FOUND)
            message(FATAL_ERROR "Could not find a supported FFT library (MKL, cuFFT or FFTW)")
        endif()
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

# Internal products

enable_testing()
add_subdirectory(src)
add_subdirectory(workflows)
