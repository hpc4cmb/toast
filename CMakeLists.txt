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

# pybind11 enables IPO by default.  This can cause problems with some
# compilers.  Remove this if we find a better workaround:
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

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
if(NOT DISABLE_OPENMP)
    find_package(OpenMP)
endif()

if(OpenMP_CXX_FOUND)
    message(STATUS
        "Found OpenMP version: \"" ${OpenMP_CXX_VERSION}
        "\" spec date " ${OpenMP_CXX_SPEC_DATE}
    )
    # Allow the user to force enabling of target offload.  Some compilers support all
    # the features we need in the 5.0 spec, but return a supported version less than
    # that.
    if(NOT USE_OPENMP_TARGET)
        if(NOT DISABLE_OPENMP_TARGET)
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
    endif()
    # Build up the final list of OpenMP flags
    set(omp_opts "")
    if(USE_OPENMP_TARGET)
        message(STATUS "Enabling support for OpenMP target offload")
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "NVHPC")
            # We are doing target offload with the NVIDIA compiler
            # and need to override the flags. NOTE:  The use of "nordc"
            # is painful and needed since nvc++ cannot (as of 22.5) build
            # standalone shared loadable modules.
            #list(APPEND omp_opts -mp=gpu -gpu=nordc -cuda -cudalibs)
            list(APPEND omp_opts -mp=gpu)
        else()
            list(APPEND omp_opts ${OpenMP_CXX_FLAGS})
        endif()
        if(OPENMP_TARGET_FLAGS)
            # Extra user-specified flags
            string(REPLACE " " ";" TARGET_FLAGS_LIST "${OPENMP_TARGET_FLAGS}")
            list(APPEND omp_opts ${TARGET_FLAGS_LIST})
        endif()
    else()
        message(STATUS
            "OpenMP target offload disabled.  Force on with -DUSE_OPENMP_TARGET=TRUE"
        )
        list(APPEND omp_opts ${OpenMP_CXX_FLAGS})
        # Some compilers will forcibly try to offload...
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            # FIXME:  Only do this for specific gcc versions?  In particular,
            # older versions don't have this problem.
            list(APPEND omp_opts -foffload=disable)
        endif()
    endif()
    set(OpenMP_CXX_FLAGS ${omp_opts})
    string(JOIN " " msg_omp_opts ${omp_opts})
    message(STATUS "Using OpenMP flags:  ${msg_omp_opts}")
    message(STATUS "Using OpenMP flags (as a list):  ${OpenMP_CXX_FLAGS}")
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

# We require libFLAC >= 1.4.0 for 32bit integer support
set(USE_FLAC FALSE)
if(DISABLE_FLAC)
    message(STATUS "FLAC support disabled")
else()
    find_package(FLAC)
    if(FLAC_FOUND)
        string(REGEX REPLACE "^([0-9]+)\\.[0-9]+\\..*" "\\1" FLAC_MAJ_VERSION "${FLAC_VERSION}")
        string(REGEX REPLACE "^[0-9]+\\.([0-9]+)\\..*" "\\1" FLAC_MIN_VERSION "${FLAC_VERSION}")
        if(FLAC_MAJ_VERSION GREATER 1)
            # Future proofing
            message(STATUS "Found FLAC version ${FLAC_VERSION}, enabling support")
            set(USE_FLAC TRUE)
        else()
            if(FLAC_MIN_VERSION GREATER_EQUAL 4)
                message(STATUS "Found FLAC version ${FLAC_VERSION}, enabling support")
                set(USE_FLAC TRUE)
            endif()
        endif()
    endif()
    if(NOT USE_FLAC)
        message(STATUS "Did not find FLAC >= 1.4.0")
    endif()
endif()

find_package(Python3 REQUIRED)

# Internal products

enable_testing()
add_subdirectory(src)
add_subdirectory(workflows)
