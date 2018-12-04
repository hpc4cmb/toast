
# Add the library target

add_library(toast
    src/toast_environment.cpp
    src/toast_utils.cpp
    src/toast_math_rng.cpp
)

target_include_directories(toast PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)

if(MPI_FOUND)
    target_compile_definitions(toast PUBLIC HAVE_MPI)
endif(MPI_FOUND)

# target_link_libraries(tidas ${HDF5_C_LIBRARIES})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Hardcode this to "lib" for now, since lib32/lib64 is a pain
# to use on many systems.
#install(TARGETS toast DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS toast DESTINATION lib)
