
# Add a custom command that produces version.cpp, plus
# a dummy output that's not actually produced, in order
# to force version.cmake to always be re-run before the build
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_version.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/src/version.cpp
    COMMAND ${CMAKE_COMMAND} -P
        ${CMAKE_CURRENT_SOURCE_DIR}/version.cmake
)

# Add the library target

set(toast_SOURCES
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
)

#if(AATM_FOUND)
#    set(toast_SOURCES ${toast_SOURCES} src/toast_tod_atm.cpp)
#endif(AATM_FOUND)

add_library(toast ${toast_SOURCES})

target_include_directories(toast PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src
)

# This compiler definition makes it possible to query the serial library
# to see if we also have the MPI library.

if(MPI_FOUND)
    target_compile_definitions(toast PRIVATE HAVE_MPI=1)
endif(MPI_FOUND)

# Dependencies

if(OpenMP_CXX_FOUND)
    target_compile_options(toast PRIVATE "${OpenMP_CXX_FLAGS}")
    set_target_properties(toast PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    target_link_libraries(toast "${OpenMP_CXX_LIBRARIES}")
endif(OpenMP_CXX_FOUND)

if(LAPACK_FOUND)
    target_compile_definitions(toast PRIVATE HAVE_LAPACK=1)
    target_compile_definitions(toast PRIVATE "LAPACK_NAMES_${LAPACK_NAMES}")
    set_target_properties(toast PROPERTIES LINK_FLAGS
        "${LAPACK_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}"
    )
    target_link_libraries(toast "${LAPACK_LIBRARIES}")
    target_link_libraries(toast "${BLAS_LIBRARIES}")
endif(LAPACK_FOUND)

if(AATM_FOUND)
    target_compile_definitions(toast PRIVATE HAVE_AATM=1)
    target_include_directories(toast PUBLIC "${AATM_INCLUDE_DIRS}")
    target_link_libraries(toast "${AATM_LIBRARIES}")
endif(AATM_FOUND)

if(FFTW_FOUND)
    target_compile_definitions(toast PRIVATE HAVE_FFTW=1)
    target_include_directories(toast PUBLIC "${FFTW_INCLUDE_DIRS}")
    target_link_libraries(toast "${FFTW_LIBRARIES}")
    if(FFTW_DOUBLE_THREADS_LIB_FOUND)
        target_compile_definitions(toast PRIVATE HAVE_FFTW_THREADS=1)
    endif(FFTW_DOUBLE_THREADS_LIB_FOUND)
endif(FFTW_FOUND)

# FIXME: restore MKL linking here after testing on a machine with Intel
# compilers.


# Installation

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Hardcode this to "lib" for now, since lib32/lib64 is a pain
# to use on many systems.
#install(TARGETS toast DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS toast DESTINATION lib)
