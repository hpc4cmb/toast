# This directory defines an object library with all the 
# external dependencies.

# Version file
set(c_version ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)

add_custom_command(OUTPUT ${c_version}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_version_cpp.sh ${SKBUILD_PROJECT_VERSION}
    COMMENT "Updating version files if needed ..."
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Define the sources

set(toast_SOURCES
    toast.cpp
    "${c_version}"
    src/toast_gpu_helpers.cpp
    src/toast_sys_environment.cpp
    src/toast_sys_utils.cpp
    src/toast_math_linearalgebra.cpp
    src/toast_math_sf.cpp
    src/toast_math_rng.cpp
    src/toast_math_qarray.cpp
    src/toast_math_fft_fftw.cpp
    src/toast_math_fft_mkl.cpp
    src/toast_math_fft_cufft.cpp
    src/toast_math_fft.cpp
    src/toast_map_cov.cpp
    src/toast_fod_psd.cpp
    src/toast_tod_filter.cpp
    src/toast_tod_simnoise.cpp
    src/toast_atm_utils.cpp
    src/toast_atm.cpp
    src/toast_atm_sim.cpp
    src/toast_atm_observe.cpp
    src/toast_template_offset.cpp
)

add_library(toast OBJECT ${toast_SOURCES})

target_include_directories(toast BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_include_directories(toast BEFORE PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Dependencies

if(AATM_FOUND)
    target_compile_definitions(toast PRIVATE HAVE_AATM=1)
    target_include_directories(toast PUBLIC "${AATM_INCLUDE_DIRS}")
    target_link_libraries(toast "${AATM_LIBRARIES}")
endif(AATM_FOUND)

if(CHOLMOD_FOUND)
    target_compile_definitions(toast PRIVATE HAVE_CHOLMOD=1)
    target_include_directories(toast PUBLIC "${CHOLMOD_INCLUDE_DIR}")
    target_link_libraries(toast
        ${CHOLMOD_LIBRARY}
        ${CCOLAMD_LIBRARY}
        ${CAMD_LIBRARY}
        ${COLAMD_LIBRARY}
        ${AMD_LIBRARY}
        ${SUITESPARSE_CONFIG_LIBRARY}
    )
    if (METIS_FOUND)
        target_link_libraries(toast ${METIS_LIBRARY})
    endif (METIS_FOUND)
endif(CHOLMOD_FOUND)

# Link to our FFT libraries

if(MKL_FOUND)
    target_compile_definitions(toast PRIVATE HAVE_MKL=1)
    target_include_directories(toast PUBLIC "${MKL_INCLUDE_DIRS}")
    target_link_libraries(toast "${MKL_LIBRARIES}")
    # If we are using GNU compilers, we will need to internally set the threading layer
    # to use GNU.  Otherwise, assume we are using Intel threads.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_definitions(toast PRIVATE USE_MKL_GNU_THREADS=1)
    else()
        target_compile_definitions(toast PRIVATE USE_MKL_INTEL_THREADS=1)
    endif()
endif(MKL_FOUND)

if(FFTW_FOUND)
    target_compile_definitions(toast PRIVATE HAVE_FFTW=1)
    target_include_directories(toast PUBLIC "${FFTW_INCLUDE_DIRS}")
    if(FFTW_DOUBLE_OPENMP_LIB_FOUND)
        target_compile_definitions(toast PRIVATE HAVE_FFTW_THREADS=1)
        target_link_libraries(toast "${FFTW_DOUBLE_OPENMP_LIB}")
    else()
        if(FFTW_DOUBLE_THREADS_LIB_FOUND)
            message(WARNING "OpenMP version of FFTW not found.  Using pthreads")
            message(WARNING "version which may break thread affinity in slurm jobs.")
            target_compile_definitions(toast PRIVATE HAVE_FFTW_THREADS=1)
            target_link_libraries(toast "${FFTW_DOUBLE_THREADS_LIB}")
        endif()
    endif()
    target_link_libraries(toast "${FFTW_DOUBLE_LIB}")
endif(FFTW_FOUND)

# LAPACK / BLAS

if(LAPACK_FOUND)
    target_compile_definitions(toast PRIVATE HAVE_LAPACK=1)
    target_compile_definitions(toast PRIVATE "LAPACK_NAMES_${LAPACK_NAMES}")
    target_link_libraries(toast "${LAPACK_LIBRARIES}")
    target_link_libraries(toast "${BLAS_LIBRARIES}")
endif(LAPACK_FOUND)

if(OpenMP_CXX_FOUND)
    target_compile_options(toast PRIVATE "${OpenMP_CXX_FLAGS}")
    target_link_libraries(toast "${OpenMP_CXX_LIBRARIES}")
endif(OpenMP_CXX_FOUND)

# CUDA

# https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html#module:FindCUDAToolkit
if(CUDAToolkit_FOUND AND NOT CUDA_DISABLED)
    message(STATUS "CUDA libs found.")
    target_compile_definitions(toast PRIVATE HAVE_CUDALIBS=1)
    target_include_directories(toast PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_include_directories(toast PRIVATE "${CUDAToolkit_MATH_INCLUDE_DIRS}")
    # there are also static versions of those libs
    target_link_libraries(toast CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::cufft CUDA::cufftw)
else()
    message(STATUS "CUDA libs not found.")
endif()

