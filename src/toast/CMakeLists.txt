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
    _libtoast/common.cpp
    _libtoast/module.cpp
    _libtoast/sys.cpp
    _libtoast/math_misc.cpp
    _libtoast/math_sf.cpp
    _libtoast/math_rng.cpp
    _libtoast/math_qarray.cpp
    _libtoast/math_fft.cpp
    _libtoast/math_healpix.cpp
    _libtoast/fod_psd.cpp
    _libtoast/tod_filter.cpp
    _libtoast/tod_pointing.cpp
    _libtoast/tod_simnoise.cpp
    _libtoast/todmap_scanning.cpp
    _libtoast/map_cov.cpp
    _libtoast/pixels.cpp
    _libtoast/todmap_mapmaker.cpp
    _libtoast/atm.cpp
    _libtoast/template_offset.cpp
    _libtoast/accelerator.cpp
)

# Handle recursion limit on NVC++ / PGI
if (CMAKE_CXX_COMPILER_ID STREQUAL PGI)
    target_compile_options(_libtoast PRIVATE -Wc,--pending_instantiations=0)
endif()
if (CMAKE_CXX_COMPILER_ID STREQUAL NVHPC)
    target_compile_options(_libtoast PRIVATE -Wc,--pending_instantiations=0)
endif()

if(OpenACC_CXX_FOUND)
    target_compile_definitions(_libtoast PRIVATE HAVE_OPENACC=1)
endif(OpenACC_CXX_FOUND)

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

if(CUDAToolkit_FOUND AND NOT CUDA_DISABLED)
    target_compile_definitions(toast PRIVATE HAVE_CUDALIBS=1)
    target_include_directories(toast PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    target_include_directories(toast PRIVATE "${CUDAToolkit_MATH_INCLUDE_DIRS}")
endif()

# Include path to the toast headers
target_include_directories(_libtoast BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/_libtoast"
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
    config.py
    job.py
    pixels.py
    pixels_io.py
    covariance.py
    dist.py
    data.py
    intervals.py
    instrument.py
    instrument_sim.py
    coordinates.py
    noise.py
    atm.py
    dipole.py
    noise_sim.py
    observation.py
    observation_dist.py
    observation_data.py
    observation_view.py
    vis.py
    rng.py
    qarray.py
    fft.py
    healpix.py
    weather.py
    schedule.py
    schedule_sim_ground.py
    schedule_sim_satellite.py
    "RELEASE"
    DESTINATION ${PYTHON_SITE}/toast
)

# Install package data

install(DIRECTORY
    aux
    DESTINATION ${PYTHON_SITE}/toast
)

# Process the sub directories
add_subdirectory(io)
add_subdirectory(tests)
add_subdirectory(ops)
add_subdirectory(templates)
add_subdirectory(scripts)
add_subdirectory(spt3g)
