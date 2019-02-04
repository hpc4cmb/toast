
set(toast_mpi_SOURCES
    src/toast_mpi.cpp
    src/toast_mpi_shmem.cpp
)

add_library(toast_mpi ${toast_mpi_SOURCES})

target_include_directories(toast_mpi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)

target_include_directories(toast_mpi PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast/include"
)

target_include_directories(toast_mpi PRIVATE "${MPI_CXX_INCLUDE_PATH}")

string(STRIP "${MPI_CXX_COMPILE_FLAGS}" mpi_comp_flags)
target_compile_options(toast_mpi PRIVATE "${mpi_comp_flags}")

string(STRIP "${MPI_CXX_LINK_FLAGS}" mpi_link_flags)
set_target_properties(toast_mpi PROPERTIES LINK_FLAGS "${mpi_link_flags}")

target_link_libraries(toast_mpi toast "${MPI_CXX_LIBRARIES}")

install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# Hardcode this to "lib" for now, since lib32/lib64 is a pain
# to use on many systems.
#install(TARGETS toast_mpi DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(TARGETS toast_mpi DESTINATION lib)
