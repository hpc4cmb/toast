
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

# Link to the libtoast object library
target_link_libraries(_libtoast PRIVATE toast)

# Include path to the toast headers
target_include_directories(_libtoast BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/_libtoast"
    "${CMAKE_CURRENT_SOURCE_DIR}/../libtoast/include"
)

# Add dependency linking flags to the final extension

if(AATM_FOUND)
    target_compile_definitions(_libtoast PUBLIC HAVE_AATM=1)
    target_include_directories(_libtoast PUBLIC "${AATM_INCLUDE_DIRS}")
    target_link_libraries(_libtoast PUBLIC "${AATM_LIBRARIES}")
endif(AATM_FOUND)

if(CHOLMOD_FOUND)
    target_compile_definitions(_libtoast PUBLIC HAVE_CHOLMOD=1)
    target_include_directories(_libtoast PUBLIC "${CHOLMOD_INCLUDE_DIR}")
    target_link_libraries(_libtoast PUBLIC
        ${CHOLMOD_LIBRARY}
        ${CCOLAMD_LIBRARY}
        ${CAMD_LIBRARY}
        ${COLAMD_LIBRARY}
        ${AMD_LIBRARY}
        ${SUITESPARSE_CONFIG_LIBRARY}
    )
    if (METIS_FOUND)
        target_link_libraries(_libtoast PUBLIC ${METIS_LIBRARY})
    endif (METIS_FOUND)
endif(CHOLMOD_FOUND)

# Link to our FFT libraries

if(MKL_FOUND)
    target_compile_definitions(_libtoast PRIVATE HAVE_MKL=1)
    target_include_directories(_libtoast PUBLIC "${MKL_INCLUDE_DIRS}")
    target_link_libraries(_libtoast PUBLIC "${MKL_LIBRARIES}")
    # If we are using GNU compilers, we will need to internally set the threading layer
    # to use GNU.  Otherwise, assume we are using Intel threads.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_definitions(_libtoast PRIVATE USE_MKL_GNU_THREADS=1)
    else()
        target_compile_definitions(_libtoast PRIVATE USE_MKL_INTEL_THREADS=1)
    endif()
endif(MKL_FOUND)

if(FFTW_FOUND)
    target_compile_definitions(_libtoast PRIVATE HAVE_FFTW=1)
    target_include_directories(_libtoast PUBLIC "${FFTW_INCLUDE_DIRS}")
    if(FFTW_DOUBLE_OPENMP_LIB_FOUND)
        target_compile_definitions(_libtoast PRIVATE HAVE_FFTW_THREADS=1)
        target_link_libraries(_libtoast PUBLIC "${FFTW_DOUBLE_OPENMP_LIB}")
    else()
        if(FFTW_DOUBLE_THREADS_LIB_FOUND)
            message(WARNING "OpenMP version of FFTW not found.  Using pthreads")
            message(WARNING "version which may break thread affinity in slurm jobs.")
            target_compile_definitions(_libtoast PRIVATE HAVE_FFTW_THREADS=1)
            target_link_libraries(_libtoast PUBLIC "${FFTW_DOUBLE_THREADS_LIB}")
        endif()
    endif()
    target_link_libraries(_libtoast PUBLIC "${FFTW_DOUBLE_LIB}")
endif(FFTW_FOUND)

# LAPACK / BLAS

if(LAPACK_FOUND)
    target_compile_definitions(_libtoast PRIVATE HAVE_LAPACK=1)
    target_compile_definitions(_libtoast PRIVATE "LAPACK_NAMES_${LAPACK_NAMES}")
    if(TOAST_STATIC_DEPS)
        target_link_libraries(_libtoast PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LAPACK_LIBRARIES}>")
        target_link_libraries(_libtoast PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${BLAS_LIBRARIES}>")
    else()
        target_link_libraries(_libtoast PUBLIC "${LAPACK_LIBRARIES}")
        target_link_libraries(_libtoast PUBLIC "${BLAS_LIBRARIES}")
    endif()
endif(LAPACK_FOUND)

if(OpenMP_CXX_FOUND)
    target_compile_options(_libtoast PRIVATE "${OpenMP_CXX_FLAGS}")
    target_link_libraries(_libtoast PUBLIC "${OpenMP_CXX_LIBRARIES}")
endif(OpenMP_CXX_FOUND)


install(TARGETS _libtoast LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

# Manually install the generated version file
install(FILES ${py_version} DESTINATION ${SKBUILD_PROJECT_NAME})

# Python files and submodules are configured in the top-level pyproject.toml.
