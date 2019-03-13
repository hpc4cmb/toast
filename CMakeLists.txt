# TOAST

# This minimum version is mostly set in order to get a newer version
# of the FindMPI check.  Note that you can easily install a newer cmake version
# using conda or pip.
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

foreach(policy
    CMP0048
    CMP0074
    CMP0077
  )
  if(POLICY ${policy})
    cmake_policy(SET ${policy} NEW)
  endif()
endforeach()

project(toast VERSION 2.3.0 LANGUAGES C CXX)

# Auxiliary files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Copy the RELEASE file into the build directory
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/RELEASE"
    "${CMAKE_CURRENT_BINARY_DIR}/RELEASE"
    COPYONLY
)

# Use GNUInstallDirs to install libraries into correct locations on all
# platforms.
include(GNUInstallDirs)

# Build defaults
include(BuildType)

# External packages

find_package(OpenMP)

find_package(BLAS)
if(BLAS_FOUND)
    find_package(LAPACK)
    find_package(LAPACKnames)
endif(BLAS_FOUND)

find_package(FFTW)

find_package(AATM)

find_package(SuiteSparse)

find_package(MPI)

find_package(PythonInterp REQUIRED)

find_package(MPI4PY)

# Internal products

enable_testing()
add_subdirectory(src)
