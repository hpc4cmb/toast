
# MPI compile flags- strip whitespace
string(STRIP "${MPI_CXX_COMPILE_FLAGS}" mpi_comp_flags)
string(STRIP "${MPI_CXX_LINK_FLAGS}" mpi_link_flags)


# Define the sources

set(toast_mpi_SOURCES
    toast_mpi.cpp
    src/toast_mpi.cpp
    tests/toast_mpi_test_runner.cpp
    tests/toast_mpi_test_shmem.cpp
)

add_library(toast_mpi ${toast_mpi_SOURCES})

target_include_directories(toast_mpi BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast"
    "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast/tests"
    "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast/gtest/googletest/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
)

target_include_directories(toast_mpi BEFORE PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(toast_mpi PRIVATE "${MPI_CXX_INCLUDE_PATH}")

target_compile_options(toast_mpi PRIVATE "${mpi_comp_flags}")

set_target_properties(toast_mpi PROPERTIES LINK_FLAGS "${mpi_link_flags}")

target_link_libraries(toast_mpi toast "${MPI_CXX_LIBRARIES}")

if(OpenMP_CXX_FOUND)
    target_compile_options(toast_mpi PRIVATE "${OpenMP_CXX_FLAGS}")
    set_target_properties(toast_mpi PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    target_link_libraries(toast_mpi "${OpenMP_CXX_LIBRARIES}")
endif(OpenMP_CXX_FOUND)

# Public headers
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# Hardcode this to "lib" for now, since lib32/lib64 is a pain
# to use on many systems.
#install(TARGETS toast_mpi DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(TARGETS toast_mpi DESTINATION lib)


# MPI unit test executable

add_executable(toast_mpi_test
    toast_mpi_test.cpp
)

if(OpenMP_CXX_FOUND)
    target_compile_options(toast_mpi_test PRIVATE "${OpenMP_CXX_FLAGS}")
    set_target_properties(toast_mpi_test PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
endif(OpenMP_CXX_FOUND)

target_include_directories(toast_mpi_test BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast"
    "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast/tests"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
)

target_include_directories(toast_mpi_test PRIVATE "${MPI_CXX_INCLUDE_PATH}")

target_compile_options(toast_mpi_test PRIVATE "${mpi_comp_flags}")

target_link_libraries(toast_mpi_test toast_mpi)

install(TARGETS toast_mpi_test DESTINATION ${CMAKE_INSTALL_BINDIR})

add_test(NAME mpi_tests COMMAND toast_mpi_test)
