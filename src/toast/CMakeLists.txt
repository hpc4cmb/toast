# pybind11
add_subdirectory(pybind11)

# Copy the RELEASE file into the build directory
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/RELEASE"
    "${CMAKE_CURRENT_BINARY_DIR}/RELEASE"
    COPYONLY
)

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
    _libtoast_pixels.cpp
    _libtoast_todmap_mapmaker.cpp
    _libtoast_atm.cpp
)

if(OpenMP_CXX_FOUND)
    target_compile_options(_libtoast PRIVATE "${OpenMP_CXX_FLAGS}")
    set_target_properties(_libtoast PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    set(OMP_LIBS "${OpenMP_CXX_LIBRARIES}")
else()
    set(OMP_LIBS "")
endif(OpenMP_CXX_FOUND)

target_link_libraries(_libtoast PRIVATE toast "${OMP_LIBS}")

if(AATM_FOUND)
    target_compile_definitions(_libtoast PRIVATE HAVE_AATM=1)
    target_include_directories(_libtoast PUBLIC "${AATM_INCLUDE_DIRS}")
endif(AATM_FOUND)

if(CHOLMOD_FOUND)
    target_compile_definitions(_libtoast PRIVATE HAVE_CHOLMOD=1)
    target_include_directories(_libtoast PUBLIC "${CHOLMOD_INCLUDE_DIR}")
endif(CHOLMOD_FOUND)

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
    traits.py
    cache.py
    config.py
    pixels.py
    pixels_io.py
    covariance.py
    cuda.py
    dist.py
    data.py
    intervals.py
    instrument.py
    instrument_sim.py
    noise.py
    noise_sim.py
    observation.py
    observation_data.py
    observation_view.py
    operator.py
    vis.py
    rng.py
    qarray.py
    fft.py
    healpix.py
    weather.py
    schedule.py
    spt3g.py
    "RELEASE"
    DESTINATION ${PYTHON_SITE}/toast
)

# Process the sub directories
add_subdirectory(tests)
add_subdirectory(tod)
add_subdirectory(map)
add_subdirectory(todmap)
add_subdirectory(fod)
add_subdirectory(pipeline_tools)
add_subdirectory(future_ops)
