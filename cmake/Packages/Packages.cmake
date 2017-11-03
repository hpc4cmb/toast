# find packages for TOAST

include(MacroUtilities)
include(GenericCMakeOptions)
include(GenericCMakeFunctions)
include(Compilers)

if(NOT CMAKE_VERSION VERSION_LESS 2.6.3)
    cmake_policy(SET CMP0011 NEW)
endif()

# - MPI
# - OpenMP
# - Python
# - BLAS
# - LAPACK
# - OpenBLAS
# - MKL
# - TBB
# - FFTW
# - wcslib
# - Elemental

add_option(USE_SSE "Use SSE/AVX optimization flags" OFF)
add_option(USE_OPENMP "Use OpenMP" ON)

add_option(USE_MKL "Enable Intel Math Kernel Library (MKL)" ${CMAKE_CXX_COMPILER_IS_INTEL})
add_option(USE_TBB "Enable Intel Thread Building Blocks (TBB)" OFF)
add_option(USE_MATH "Enable Intel IMF Math library" ${CMAKE_CXX_COMPILER_IS_INTEL})

add_option(USE_BLAS "Use BLAS" ON)
add_option(USE_LAPACK "Use LAPACK" ON)
#add_option(USE_OPENBLAS "Use OpenBLAS" OFF)

add_option(USE_FFTW "Use FFTW" OFF)
add_option(USE_WCSLIB "Use wcslib" OFF)
add_option(USE_ELEMENTAL "Use Elemental" OFF)

if(APPLE AND CMAKE_CXX_COMPILER_IS_GNU)
    add_option(USE_CLANG_ASSEMBLER
        "SSE/AVX GNU compiler errors (no such instruction) on macOS might be fixed by turning this on"
        OFF)
endif(APPLE AND CMAKE_CXX_COMPILER_IS_GNU)


################################################################################
#
#        MPI
#
################################################################################

ConfigureRootSearchPath(MPI)
find_package(MPI REQUIRED)

# Add the MPI-specific compiler and linker flags
add(CMAKE_C_FLAGS_EXTRA    "${MPI_C_COMPILE_FLAGS}")
add(CMAKE_CXX_FLAGS_EXTRA  "${MPI_CXX_COMPILE_FLAGS}")
add(CMAKE_EXE_LINKER_FLAGS "${MPI_CXX_LINK_FLAGS}")
add_definitions(-DUSE_MPI)
add_definitions(-DHAVE_CONFIG_H)

set(MPI_LIBRARIES )
foreach(_TYPE C_LIBRARIES CXX_LIBRARIES EXTRA_LIBRARY)
    set(_TYPE MPI_${_TYPE})
    if(${_TYPE})
        list(APPEND MPI_LIBRARIES ${${_TYPE}})
    endif(${_TYPE})
endforeach(_TYPE C_LIBRARIES CXX_LIBRARIES EXTRA_LIBRARY)


################################################################################
#
#        Python
#
################################################################################

find_package( PythonInterp 3.3 REQUIRED )
find_package( PythonLibs 3.3 REQUIRED )

find_package_handle_standard_args(Python3 DEFAULT_MSG
    PYTHON_VERSION_STRING PYTHON_INCLUDE_DIRS
    PYTHON_LIBRARIES)


################################################################################
#
#        OpenMP
#
################################################################################
if(USE_OPENMP)

    if(CMAKE_CXX_COMPILER_IS_CLANG AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        message(AUTHOR_WARNING
            "Clang on Darwin (as of OS X Mavericks) does not have OpenMP Support")
    endif()

    find_package(OpenMP REQUIRED)
    # Add the OpenMP-specific compiler and linker flags
    add(CMAKE_C_FLAGS_EXTRA   "${OpenMP_C_FLAGS}")
    add(CMAKE_CXX_FLAGS_EXTRA "${OpenMP_CXX_FLAGS}")
    add_definitions(-DUSE_OPENMP)

endif(USE_OPENMP)


################################################################################
#
#        MKL - Intel Math Kernel Library
#
################################################################################
if(USE_MKL)

    ConfigureRootSearchPath(MKL)

    set(MKL_COMPONENTS rt) # mkl_rt as default

    if(CMAKE_C_COMPILER_IS_GNU OR CMAKE_CXX_COMPILER_IS_GNU)
        set_ifnot(MKL_THREADING OpenMP)
        set(MKL_COMPONENTS gnu_thread core intel_lp64)
        find_package(GOMP REQUIRED)
    endif()

    find_package(MKL REQUIRED COMPONENTS ${MKL_COMPONENTS})

    foreach(_def ${MKL_DEFINITIONS})
        add_definitions(-D${_def})
    endforeach()
    add(EXTERNAL_LINK_FLAGS "${MKL_CXX_LINK_FLAGS}")
    add_definitions(-DUSE_MKL)

endif()


################################################################################
#
#        TBB - Intel Thread Building Blocks
#
################################################################################
if(USE_TBB)

    ConfigureRootSearchPath(TBB)
    find_package(TBB REQUIRED COMPONENTS malloc)

    add_definitions(-DUSE_TBB)
    if(TBB_MALLOC_FOUND)
        add_definitions(-DUSE_TBB_MALLOC)
    endif()

endif()


################################################################################
#
#        Math - Intel IMF library
#
################################################################################
if(USE_MATH)

    ConfigureRootSearchPath(IMF)
    find_package(IMF REQUIRED)

    add_definitions(-DUSE_MATH_IMF)

endif()


################################################################################
#
#        BLAS, LAPACK, and OpenBLAS
#
################################################################################
foreach(_LIB BLAS LAPACK OpenBLAS)

    string(TOUPPER "${_LIB}" _UPPER_LIB)
    if(USE_${_UPPER_LIB})

        ConfigureRootSearchPath(${_LIB})
        find_package(${_LIB})

    endif()

    if(${_LIB}_FOUND)
        add_definitions(-DUSE_${_UPPER_LIB})
    else(${_LIB}_FOUND)
        unset(${_LIB}_LIBRARIES)
    endif(${_LIB}_FOUND)

endforeach()


################################################################################
#
#        FFTW
#
################################################################################
if(USE_FFTW)

    ConfigureRootSearchPath(FFTW3)
    find_package(FFTW3 REQUIRED)

    if(NOT USE_MKL)
        # double precision with threads
        find_package(FFTW3 COMPONENTS threads)
    endif()

    add_definitions(-DUSE_FFTW)

endif(USE_FFTW)


################################################################################
#
#        wcslib
#
################################################################################
if(USE_WCSLIB)

    ConfigureRootSearchPath(wcslib)
    find_package(wcslib REQUIRED)

    add_definitions(-DUSE_WCSLIB)

endif(USE_WCSLIB)


################################################################################
#
#        Elemental
#
################################################################################
if(USE_ELEMENTAL)

    ConfigureRootSearchPath(Elemental)
    find_package(Elemental REQUIRED)

    add_definitions(-DUSE_ELEMENTAL)

endif(USE_ELEMENTAL)


################################################################################
# --- setting definitions: EXTERNAL_INCLUDE_DIRS,   ---------------------------#
#                          EXTERNAL_LIBRARIES       ---------------------------#
################################################################################

set(EXTERNAL_INCLUDE_DIRS
    ${MPI_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH}    
    ${MKL_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
    ${IMF_INCLUDE_DIRS}
    ${FFTW3_INCLUDE_DIRS}
    ${wcslib_INCLUDE_DIRS}
    ${Elemental_INCLUDE_DIRS}
)

set(EXTERNAL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT}
    ${MPI_LIBRARIES}
    ${MKL_LIBRARIES}
    ${GOMP_LIBRARIES}
    ${TBB_LIBRARIES}
    ${IMF_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${OpenBLAS_LIBRARIES}
    ${FFTW3_LIBRARIES}
    ${wcslib_LIBRARIES}
    ${Elemental_LIBRARIES}
)

REMOVE_DUPLICATES(EXTERNAL_INCLUDE_DIRS)
REMOVE_DUPLICATES(EXTERNAL_LIBRARIES)

################################################################################
#
#        SSE
#
################################################################################

if(USE_SSE)

    include(FindSSE)

    GET_SSE_COMPILE_FLAGS(_CXX_FLAGS_EXTRA SSE_DEFINITIONS)
    foreach(_DEF ${SSE_DEFINITIONS})
        add_definitions(-D${_DEF})
    endforeach()
    unset(SSE_DEFINITIONS)

    add(CMAKE_CXX_FLAGS_EXTRA "${SSE_FLAGS}")

else(USE_SSE)

    foreach(type SSE2 SSE3 SSSE3 SSE4_1 AVX AVX2)
        remove_definitions(-DHAS_${type})
    endforeach()

endif(USE_SSE)

################################################################################
#
#        Architecture Flags
#
################################################################################

include(Architecture)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" OR
   "${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo" OR
   DEFINED TARGET_ARCHITECTURE)

    ArchitectureFlags(ARCH_FLAGS)
    add(CMAKE_C_FLAGS_EXTRA "${ARCH_FLAGS}")
    add(CMAKE_CXX_FLAGS_EXTRA "${ARCH_FLAGS}")

endif()
