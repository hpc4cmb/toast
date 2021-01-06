
# Add internal googletest framework.  Must be added before the unit test
# internal library in order to define some variables.

add_subdirectory(gtest EXCLUDE_FROM_ALL)
set(gtest_force_shared_crt TRUE)
include_directories("${gtest_SOURCE_DIR}/include")

# Add a custom command that produces version.cpp

set(versioncpp ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)

add_custom_command(OUTPUT ${versioncpp}
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/version.cmake
    COMMENT "Updating version.cpp if needed ..."
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Define the sources

set(toast_SOURCES
    toast.cpp
    src/toast_sys_environment.cpp
    src/toast_sys_utils.cpp
    src/toast_math_lapack.cpp
    src/toast_math_sf.cpp
    src/toast_math_rng.cpp
    src/toast_math_qarray.cpp
    src/toast_math_fft.cpp
    src/toast_math_healpix.cpp
    src/toast_map_cov.cpp
    src/toast_fod_psd.cpp
    src/toast_tod_filter.cpp
    src/toast_tod_pointing.cpp
    src/toast_tod_simnoise.cpp
    src/toast_atm_utils.cpp
    src/toast_atm.cpp
    src/toast_atm_sim.cpp
    src/toast_atm_observe.cpp
    src/toast_template_offset.cpp
    tests/toast_test_runner.cpp
    tests/toast_test_env.cpp
    tests/toast_test_utils.cpp
    tests/toast_test_sf.cpp
    tests/toast_test_rng.cpp
    tests/toast_test_qarray.cpp
    tests/toast_test_fft.cpp
    tests/toast_test_healpix.cpp
    tests/toast_test_cov.cpp
    tests/toast_test_polyfilter.cpp
)

add_library(toast OBJECT ${versioncpp} ${toast_SOURCES})

target_include_directories(toast BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
)

target_include_directories(toast BEFORE PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# This compiler definition makes it possible to query the serial library
# to see if we also have the MPI library.

if(MPI_FOUND)
    target_compile_definitions(toast PRIVATE HAVE_MPI=1)
endif(MPI_FOUND)

if(MPI4PY_FOUND)
    target_compile_definitions(toast PRIVATE HAVE_MPI4PY=1)
endif(MPI4PY_FOUND)

# Dependencies

target_link_libraries(toast gtest)

if(OpenMP_CXX_FOUND)
    target_compile_options(toast PRIVATE "${OpenMP_CXX_FLAGS}")
    set_target_properties(toast PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    target_link_libraries(toast "${OpenMP_CXX_LIBRARIES}")
endif(OpenMP_CXX_FOUND)

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
    target_link_libraries(toast "${FFTW_LIBRARIES}")
    if(FFTW_DOUBLE_THREADS_LIB_FOUND)
        target_compile_definitions(toast PRIVATE HAVE_FFTW_THREADS=1)
    endif(FFTW_DOUBLE_THREADS_LIB_FOUND)
endif(FFTW_FOUND)

# LAPACK / BLAS

if(LAPACK_FOUND)
    target_compile_definitions(toast PRIVATE HAVE_LAPACK=1)
    target_compile_definitions(toast PRIVATE "LAPACK_NAMES_${LAPACK_NAMES}")
    set_target_properties(toast PROPERTIES LINK_FLAGS
        "${LAPACK_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}"
    )
    target_link_libraries(toast "${LAPACK_LIBRARIES}")
    target_link_libraries(toast "${BLAS_LIBRARIES}")
endif(LAPACK_FOUND)

# Installation

#install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Hardcode this to "lib" for now, since lib32/lib64 is a pain
# to use on many systems.
#install(TARGETS toast DESTINATION ${CMAKE_INSTALL_LIBDIR})
#install(TARGETS toast DESTINATION lib)

# Add the serial unit test executable

add_executable(toast_test
    toast_test.cpp
)

if(OpenMP_CXX_FOUND)
    target_compile_options(toast_test PRIVATE "${OpenMP_CXX_FLAGS}")
    set_target_properties(toast_test PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
endif(OpenMP_CXX_FOUND)

target_include_directories(toast_test BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
)

target_link_libraries(toast_test toast)

install(TARGETS toast_test DESTINATION ${CMAKE_INSTALL_BINDIR})

add_test(NAME serial_tests COMMAND toast_test)
