
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(libtidas)

add_subdirectory(python)

add_subdirectory(tests)

add_subdirectory(libtoast-mpi)

add_subdirectory(tests-mpi)
