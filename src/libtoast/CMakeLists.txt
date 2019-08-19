
# Add a custom command that produces version.cpp, plus
# a dummy output that's not actually produced, in order
# to force version.cmake to always be re-run before the build
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_version.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/src/version.cpp
    COMMAND ${CMAKE_COMMAND} -P
        ${CMAKE_CURRENT_SOURCE_DIR}/version.cmake
)

# Define the core internal lbirary

set(toastcore_SOURCES
    src/version.cpp
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
)

add_library(toastcore ${toastcore_SOURCES})

target_include_directories(toastcore BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# This compiler definition makes it possible to query the serial library
# to see if we also have the MPI library.

if(MPI_FOUND)
    target_compile_definitions(toastcore PRIVATE HAVE_MPI=1)
endif(MPI_FOUND)

# Dependencies

if(OpenMP_CXX_FOUND)
    target_compile_options(toastcore PRIVATE "${OpenMP_CXX_FLAGS}")
    set_target_properties(toastcore PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    target_link_libraries(toastcore "${OpenMP_CXX_LIBRARIES}")
endif(OpenMP_CXX_FOUND)

if(AATM_FOUND)
    target_compile_definitions(toastcore PRIVATE HAVE_AATM=1)
    target_include_directories(toastcore PUBLIC "${AATM_INCLUDE_DIRS}")
    target_link_libraries(toastcore "${AATM_LIBRARIES}")
endif(AATM_FOUND)

if(CHOLMOD_FOUND)
    target_compile_definitions(toastcore PRIVATE HAVE_CHOLMOD=1)
    target_include_directories(toastcore PUBLIC "${CHOLMOD_INCLUDE_DIR}")
    target_link_libraries(toastcore "${CHOLMOD_LIBRARY}")
endif(CHOLMOD_FOUND)

if(FFTW_FOUND)
    target_compile_definitions(toastcore PRIVATE HAVE_FFTW=1)
    target_include_directories(toastcore PUBLIC "${FFTW_INCLUDE_DIRS}")
    target_link_libraries(toastcore "${FFTW_LIBRARIES}")
    if(FFTW_DOUBLE_THREADS_LIB_FOUND)
        target_compile_definitions(toastcore PRIVATE HAVE_FFTW_THREADS=1)
    endif(FFTW_DOUBLE_THREADS_LIB_FOUND)
endif(FFTW_FOUND)

if(LAPACK_FOUND)
    target_compile_definitions(toastcore PRIVATE HAVE_LAPACK=1)
    target_compile_definitions(toastcore PRIVATE "LAPACK_NAMES_${LAPACK_NAMES}")
    set_target_properties(toastcore PROPERTIES LINK_FLAGS
        "${LAPACK_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}"
    )
    target_link_libraries(toastcore "${LAPACK_LIBRARIES}")
    target_link_libraries(toastcore "${BLAS_LIBRARIES}")
endif(LAPACK_FOUND)

# FIXME: restore MKL linking here after testing on a machine with Intel
# compilers.




# Add internal googletest framework.  Must be added before the unit test
# internal library in order to define some variables.

add_subdirectory(gtest EXCLUDE_FROM_ALL)
set(gtest_force_shared_crt TRUE)
include_directories("${gtest_SOURCE_DIR}/include")

# Define the unit test internal library.  We keep this separate so that we
# can link it with different options to ensure that all unit tests are
# included in the library.

set(toasttest_SOURCES
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

add_library(toasttest ${toasttest_SOURCES})

target_include_directories(toasttest BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
)

if(OpenMP_CXX_FOUND)
    target_compile_options(toasttest PRIVATE "${OpenMP_CXX_FLAGS}")
endif(OpenMP_CXX_FOUND)

if(AATM_FOUND)
    target_compile_definitions(toasttest PRIVATE HAVE_AATM=1)
    target_include_directories(toasttest PRIVATE "${AATM_INCLUDE_DIRS}")
endif(AATM_FOUND)

if(CHOLMOD_FOUND)
    target_compile_definitions(toasttest PRIVATE HAVE_CHOLMOD=1)
    target_include_directories(toasttest PRIVATE "${CHOLMOD_INCLUDE_DIR}")
endif(CHOLMOD_FOUND)

if(FFTW_FOUND)
    target_compile_definitions(toasttest PRIVATE HAVE_FFTW=1)
    target_include_directories(toasttest PRIVATE "${FFTW_INCLUDE_DIRS}")
    if(FFTW_DOUBLE_THREADS_LIB_FOUND)
        target_compile_definitions(toasttest PRIVATE HAVE_FFTW_THREADS=1)
    endif(FFTW_DOUBLE_THREADS_LIB_FOUND)
endif(FFTW_FOUND)

if(LAPACK_FOUND)
    target_compile_definitions(toasttest PRIVATE HAVE_LAPACK=1)
    target_compile_definitions(toasttest PRIVATE "LAPACK_NAMES_${LAPACK_NAMES}")
endif(LAPACK_FOUND)

target_link_libraries(toasttest gtest)


# Define the full toast library

add_library(toast toast.cpp)

target_include_directories(toast BEFORE PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(APPLE)
    SET(TOASTTESTLIB -Wl,-force_load toasttest)
    SET(TOASTCORELIB toastcore)
#    SET(TOASTCORELIB -Wl,-all_load toastcore -Wl,-noall_load)
else()
    SET(TOASTTESTLIB -Wl,--whole-archive toasttest -Wl,--no-whole-archive)
    SET(TOASTCORELIB toastcore)
#    SET(TOASTCORELIB -Wl,--whole-archive toastcore -Wl,--no-whole-archive)
endif(APPLE)

if(OpenMP_CXX_FOUND)
    target_compile_options(toast PRIVATE "${OpenMP_CXX_FLAGS}")
    set_target_properties(toast PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
endif(OpenMP_CXX_FOUND)

target_link_libraries(toast ${TOASTTESTLIB} ${TOASTCORELIB})

# Installation

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Hardcode this to "lib" for now, since lib32/lib64 is a pain
# to use on many systems.
#install(TARGETS toast DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS toast DESTINATION lib)


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
