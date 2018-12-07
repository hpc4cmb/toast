
# Add the library target

add_library(toast
    src/toast_environment.cpp
    src/toast_utils.cpp
    src/toast_math_sf.cpp
    src/toast_math_rng.cpp
    src/toast_math_qarray.cpp
)

target_include_directories(toast PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)

if(MPI_FOUND)
    target_compile_definitions(toast PRIVATE HAVE_MPI=1)
endif(MPI_FOUND)

# Dependencies

if(OpenMP_CXX_FOUND)
    target_compile_options(toast PRIVATE ${OpenMP_CXX_FLAGS})
    set_target_properties(toast PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
    target_link_libraries(toast "${OpenMP_CXX_LIBRARIES}")
endif(OpenMP_CXX_FOUND)

# Installation

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Hardcode this to "lib" for now, since lib32/lib64 is a pain
# to use on many systems.
#install(TARGETS toast DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS toast DESTINATION lib)
