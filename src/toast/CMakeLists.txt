# pybind11
add_subdirectory(pybind11)

# Determine our python install location and set up a rule to install the
# modules

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
        import sys
        print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))"
    OUTPUT_VARIABLE PYTHON_MAJORMINOR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Hardcode this to "lib" for now, since lib32/lib64 is a pain
# to use on many systems.
#set(PYTHON_SITE
#    "${CMAKE_INSTALL_LIBDIR}/python${PYTHON_MAJORMINOR}/site-packages")
set(PYTHON_SITE "lib/python${PYTHON_MAJORMINOR}/site-packages")

# Create a module for the serial toast library

pybind11_add_module(_libtoast MODULE
    _libtoast.cpp
    _libtoast_sys.cpp
    _libtoast_math_sf.cpp
    _libtoast_math_rng.cpp
)
target_link_libraries(_libtoast PRIVATE toast)

# Include path to the toast headers
target_include_directories(_libtoast PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast/include"
)

install(TARGETS _libtoast DESTINATION ${PYTHON_SITE}/toast)

# Install all the other python files

install(FILES
    __init__.py
    utils.py
    mpi.py
    cache.py
    dist.py
    op.py
    vis.py
    DESTINATION ${PYTHON_SITE}/toast
)

# Optionally create a module for the MPI library

if(MPI_FOUND AND MPI4PY_FOUND)
    pybind11_add_module(_libtoast_mpi MODULE _libtoast_mpi.cpp)

    string(STRIP "${MPI_CXX_COMPILE_FLAGS}" mpi_comp_flags)
    target_compile_options(_libtoast_mpi PRIVATE "${mpi_comp_flags}")

    string(STRIP "${MPI_CXX_LINK_FLAGS}" mpi_link_flags)
    set_target_properties(_libtoast_mpi
        PROPERTIES LINK_FLAGS "${mpi_link_flags}"
    )

    target_include_directories(_libtoast_mpi PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast_mpi/include"
        ${MPI_CXX_INCLUDE_PATH}
        ${MPI4PY_INCLUDE_DIR}
    )

    target_link_libraries(_libtoast_mpi
        PRIVATE toast_mpi "${MPI_CXX_LIBRARIES}"
    )

    install(TARGETS _libtoast_mpi DESTINATION ${PYTHON_SITE}/toast)

    #install(FILES
    #    mpi.py
    #    DESTINATION ${PYTHON_SITE}/toast
    #)
endif(MPI_FOUND AND MPI4PY_FOUND)

# Process the sub directories
add_subdirectory(tests)
