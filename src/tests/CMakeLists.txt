
# Add internal googletest framework.  Must be added before the library
# in order to define some variables.

set(gtest_force_shared_crt TRUE)
add_subdirectory(gtest)
include_directories("${gtest_SOURCE_DIR}/include")

add_executable(toast_test
    toast_test.cpp
    toast_test_runner.cpp
    toast_test_env.cpp
    toast_test_utils.cpp
    toast_test_sf.cpp
    toast_test_rng.cpp
    toast_test_qarray.cpp
    toast_test_fft.cpp
    toast_test_healpix.cpp
    toast_test_cov.cpp
    toast_test_polyfilter.cpp
)

target_include_directories(toast_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast/include"
)

target_link_libraries(toast_test toast gtest gtest_main)

install(TARGETS toast_test DESTINATION ${CMAKE_INSTALL_BINDIR})

add_test(NAME serial_tests COMMAND toast_test)

# Optional MPI tests

if(MPI_FOUND)

    add_executable(toast_mpi_test
        toast_mpi_test.cpp
        toast_mpi_test_runner.cpp
        toast_mpi_test_shmem.cpp
    )

    target_include_directories(toast_mpi_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast_mpi/include"
    )

    target_include_directories(toast_mpi_test PRIVATE
        "${MPI_CXX_INCLUDE_PATH}"
    )

    target_link_libraries(toast_mpi_test toast_mpi gtest gtest_main)

    install(TARGETS toast_mpi_test DESTINATION ${CMAKE_INSTALL_BINDIR})

    add_test(NAME mpi_tests COMMAND toast_mpi_test)

endif(MPI_FOUND)
