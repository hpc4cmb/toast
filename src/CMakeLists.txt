
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(libtoast)
add_subdirectory(tests)

if(MPI_FOUND)
    add_subdirectory(libtoast_mpi)
endif(MPI_FOUND)

add_subdirectory(python)
