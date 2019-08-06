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
    _libtoast_common.cpp
    _libtoast.cpp
    _libtoast_sys.cpp
    _libtoast_math_sf.cpp
    _libtoast_math_rng.cpp
    _libtoast_math_qarray.cpp
    _libtoast_math_fft.cpp
    _libtoast_math_healpix.cpp
    _libtoast_fod_psd.cpp
    _libtoast_tod_filter.cpp
    _libtoast_tod_pointing.cpp
    _libtoast_tod_simnoise.cpp
    _libtoast_todmap_scanning.cpp
    _libtoast_map_cov.cpp
    _libtoast_atm.cpp
)

if(OpenMP_CXX_FOUND)
    target_compile_options(_libtoast PRIVATE "${OpenMP_CXX_FLAGS}")
    set_target_properties(_libtoast PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    set(OMP_LIBS "${OpenMP_CXX_LIBRARIES}")
else()
    set(OMP_LIBS "")
endif(OpenMP_CXX_FOUND)

if(AATM_FOUND)
    target_compile_definitions(_libtoast PRIVATE HAVE_AATM=1)
    target_include_directories(_libtoast PUBLIC "${AATM_INCLUDE_DIRS}")
endif(AATM_FOUND)

if(CHOLMOD_FOUND)
    target_compile_definitions(_libtoast PRIVATE HAVE_CHOLMOD=1)
    target_include_directories(_libtoast PUBLIC "${CHOLMOD_INCLUDE_DIR}")
endif(CHOLMOD_FOUND)

target_link_libraries(_libtoast PRIVATE toast "${OMP_LIBS}")

# Include path to the toast headers
target_include_directories(_libtoast BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast/include"
)

install(TARGETS _libtoast DESTINATION ${PYTHON_SITE}/toast)

# Install all the other python files

install(FILES
    __init__.py
    utils.py
    mpi.py
    timing.py
    cache.py
    dist.py
    op.py
    vis.py
    rng.py
    qarray.py
    fft.py
    healpix.py
    weather.py
    DESTINATION ${PYTHON_SITE}/toast
)

# Optionally create a module for the MPI library

if(MPI_FOUND AND MPI4PY_FOUND)
    pybind11_add_module(_libtoast_mpi MODULE
        _libtoast_common.cpp
        _libtoast_mpi.cpp
    )

    string(STRIP "${MPI_CXX_COMPILE_FLAGS}" mpi_comp_flags)
    target_compile_options(_libtoast_mpi PRIVATE "${mpi_comp_flags}")

    string(STRIP "${MPI_CXX_LINK_FLAGS}" mpi_link_flags)
    set_target_properties(_libtoast_mpi
        PROPERTIES LINK_FLAGS "${mpi_link_flags}"
    )

    target_include_directories(_libtoast_mpi BEFORE PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast_mpi/include"
        ${MPI_CXX_INCLUDE_PATH}
        ${MPI4PY_INCLUDE_DIR}
    )

    if(OpenMP_CXX_FOUND)
        target_compile_options(_libtoast_mpi PRIVATE "${OpenMP_CXX_FLAGS}")
        set_target_properties(_libtoast_mpi PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    endif(OpenMP_CXX_FOUND)

    if(AATM_FOUND)
        target_compile_definitions(_libtoast_mpi PRIVATE HAVE_AATM=1)
        target_include_directories(_libtoast_mpi PUBLIC "${AATM_INCLUDE_DIRS}")
    endif(AATM_FOUND)

    if(CHOLMOD_FOUND)
        target_compile_definitions(_libtoast_mpi PRIVATE HAVE_CHOLMOD=1)
        target_include_directories(_libtoast_mpi PUBLIC "${CHOLMOD_INCLUDE_DIR}")
    endif(CHOLMOD_FOUND)

    target_link_libraries(_libtoast_mpi
        PRIVATE toast_mpi "${MPI_CXX_LIBRARIES}" "${OMP_LIBS}"
    )

    install(TARGETS _libtoast_mpi DESTINATION ${PYTHON_SITE}/toast)

endif(MPI_FOUND AND MPI4PY_FOUND)

# Process the sub directories
add_subdirectory(tests)
add_subdirectory(tod)
add_subdirectory(fod)
add_subdirectory(map)
add_subdirectory(pipeline_tools)
