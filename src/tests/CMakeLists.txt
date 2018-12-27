
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
)

target_include_directories(toast_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast/src"
)

target_link_libraries(toast_test toast gtest gtest_main)

# if (HDF5_FOUND)
#     target_compile_definitions(toast_test PRIVATE HAVE_HDF5=1)
# endif (HDF5_FOUND)

install(TARGETS toast_test DESTINATION ${CMAKE_INSTALL_BINDIR})

add_test(NAME serial_tests COMMAND toast_test)
