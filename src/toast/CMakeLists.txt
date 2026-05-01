
# Version file
set(py_version ${CMAKE_CURRENT_BINARY_DIR}/_version.py)

add_custom_command(OUTPUT ${py_version}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_version_py.sh ${SKBUILD_PROJECT_VERSION}
    COMMENT "Updating version files if needed ..."
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

add_custom_target(generated_version ALL DEPENDS ${py_version})

# Create a module for the serial toast library

pybind11_add_module(_libtoast MODULE
    _libtoast/common.cpp
    _libtoast/module.cpp
    _libtoast/sys.cpp
    _libtoast/intervals.cpp
    _libtoast/math_misc.cpp
    _libtoast/math_sf.cpp
    _libtoast/math_rng.cpp
    _libtoast/math_qarray.cpp
    _libtoast/math_fft.cpp
    _libtoast/fod_psd.cpp
    _libtoast/tod_filter.cpp
    _libtoast/tod_simnoise.cpp
    _libtoast/todmap_scanning.cpp
    _libtoast/map_cov.cpp
    _libtoast/pixels.cpp
    _libtoast/ops_filterbin.cpp
    _libtoast/atm.cpp
    _libtoast/template_offset.cpp
    _libtoast/accelerator.cpp
    _libtoast/qarray_core.cpp
    _libtoast/ops_pointing_detector.cpp
    _libtoast/ops_stokes_weights.cpp
    _libtoast/ops_pixels_healpix.cpp
    _libtoast/ops_mapmaker_utils.cpp
    _libtoast/ops_noise_weight.cpp
    _libtoast/ops_scan_map.cpp
)

# Handle recursion limit on NVC++ / PGI
if (CMAKE_CXX_COMPILER_ID STREQUAL PGI)
    target_compile_options(_libtoast PRIVATE -Wc,--pending_instantiations=0)
endif()
if (CMAKE_CXX_COMPILER_ID STREQUAL NVHPC)
    target_compile_options(_libtoast PRIVATE -Wc,--pending_instantiations=0)
endif()

target_link_libraries(_libtoast PRIVATE toast)

# Include path to the toast headers
target_include_directories(_libtoast BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/_libtoast"
    "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast/include"
)

install(TARGETS _libtoast LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

# Manually install the generated version file
install(FILES ${py_version} DESTINATION ${SKBUILD_PROJECT_NAME})

# Python files and submodules are configured in the top-level pyproject.toml.
