
add_option(DASHBOARD_MODE "Internally used to skip generation of CDash files" OFF NO_FEATURE)

set_ifnot(TEST_BINDIR "${CMAKE_BINARY_DIR}" CACHE DIRECTORY 
    "Path to bin directory for testing executables")

set(CMAKE_CONFIGURE_OPTIONS )
if(NOT "${TEST_BINDIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    set(CMAKE_CONFIGURE_OPTIONS "-DTEST_BINDIR=${TEST_BINDIR}")
endif(NOT "${TEST_BINDIR}" STREQUAL "${CMAKE_BINARY_DIR}")

if(NOT DASHBOARD_MODE)
    add_option(USE_LOCAL_FOR_CTEST "Use the local source tree for CTest/CDash" OFF)
    if(USE_LOCAL_FOR_CTEST)
        set(CMAKE_LOCAL_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endif(USE_LOCAL_FOR_CTEST)
endif(NOT DASHBOARD_MODE)

# ------------------------------------------------------------------------ #
# -- Function to create a temporary directory
# ------------------------------------------------------------------------ #

function(GET_TEMPORARY_DIRECTORY DIR_VAR DIR_BASE)
    set(_TMP_ROOT "$ENV{TMPDIR}")
    if("${_TMP_ROOT}" STREQUAL "")
        set(_TMP_ROOT "/tmp")
    endif()
    find_program(MKTEMP_COMMAND NAMES mktemp)
    set(_TMP_STR "${_TMP_ROOT}/${DIR_BASE}-XXXX")
    if(MKTEMP_COMMAND)
        execute_process(COMMAND ${MKTEMP_COMMAND} -d ${_TMP_STR}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            OUTPUT_VARIABLE _VAR
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(${DIR_VAR} "${_VAR}" PARENT_SCOPE)
    else()
        message(WARNING "Unable to create temporary directory - using ${_TMP_STR}")
        execute_process(COMMAND "${CMAKE_COMMAND} -E make_directory ${_TMP_STR}")
        set(${DIR_VAR} "${_TMP_STR}" PARENT_SCOPE)
    endif(MKTEMP_COMMAND)
endfunction()

# ------------------------------------------------------------------------ #
# -- Configure Branch label
# ------------------------------------------------------------------------ #
execute_process(COMMAND git branch --contains
    OUTPUT_VARIABLE CMAKE_SOURCE_BRANCH
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "*" "" CMAKE_SOURCE_BRANCH "${CMAKE_SOURCE_BRANCH}")
string(REPLACE " " "" CMAKE_SOURCE_BRANCH "${CMAKE_SOURCE_BRANCH}")


# ------------------------------------------------------------------------ #
# -- Configure CTest for CDash
# ------------------------------------------------------------------------ #
if(NOT DASHBOARD_MODE)
    # get temporary directory for dashboard testing
    GET_TEMPORARY_DIRECTORY(CMAKE_DASHBOARD_ROOT "${CMAKE_PROJECT_NAME}-cdash")
    
    ## -- USE_<PROJECT> and <PROJECT>_ROOT
    foreach(_OPTION ARCH BLAS ELEMENTAL LAPACK OPENMP 
        PYTHON_INCLUDE_DIR SSE TBB TIMERS WCSLIB FFTW MATH MKL IMF)
        # add "USE_" definition
        if(NOT "${USE_${_OPTION}}" STREQUAL "")
            add(CMAKE_CONFIGURE_OPTIONS "-DUSE_${_OPTION}=${USE_${_OPTION}}")
        endif(NOT "${USE_${_OPTION}}" STREQUAL "")
        # add "*_ROOT" definition    
        if(NOT "${${_OPTION}_ROOT}" STREQUAL "")
            add(CMAKE_CONFIGURE_OPTIONS "-D${_OPTION}_ROOT=${${_OPTION}_ROOT}")
        endif(NOT "${${_OPTION}_ROOT}" STREQUAL "")
        # for Elemental, e.g. USE_ELEMENTAL --> Elemental_ROOT
        capitalize("${_OPTION}" C_OPTION)
        if(NOT "${${C_OPTION}_ROOT}" STREQUAL "")
            add(CMAKE_CONFIGURE_OPTIONS "-D${C_OPTION}_ROOT=${${C_OPTION}_ROOT}")
        endif(NOT "${${C_OPTION}_ROOT}" STREQUAL "")
    endforeach()
        
    ## -- CTest Config
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/CTestConfig.cmake.in
        ${CMAKE_BINARY_DIR}/CTestConfig.cmake @ONLY)
    
    ENABLE_TESTING()
    include(CTest)
    
    set(CTEST_MODEL Continuous)
    foreach(_type CDashExec CDashBuild CDashTest)
        ## -- CTest Setup
        configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/${_type}.cmake.in
            ${CMAKE_BINARY_DIR}/${_type}.cmake @ONLY)
    endforeach(_type CDashExec CDashBuild CDashTest)

    ## -- CTest Custom
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/CTestCustom.cmake.in
        ${CMAKE_BINARY_DIR}/CTestCustom.cmake @ONLY)

endif(NOT DASHBOARD_MODE)

# ------------------------------------------------------------------------ #
# -- Configure CTest tests
# ------------------------------------------------------------------------ #
ENABLE_TESTING()
    
add_test(NAME cxx_toast_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${TEST_BINDIR}/toast_test)
set_tests_properties(cxx_toast_test PROPERTIES LABELS "UnitTest;CXX" TIMEOUT 7200)

set(_PYDIR "${CMAKE_SOURCE_DIR}/src/python/tests")
FILE(GLOB _PYTHON_TEST_FILES "${_PYDIR}/*.py")
LIST(REMOVE_ITEM _PYTHON_TEST_FILES ${_PYDIR}/__init__.py ${_PYDIR}/runner.py)
set(PYTHON_TEST_FILES )
foreach(_FILE ${_PYTHON_TEST_FILES})
    get_filename_component(_FILE_NE "${_FILE}" NAME_WE)
    list(APPEND PYTHON_TEST_FILES "${_FILE_NE}")
endforeach(_FILE ${_PYTHON_TEST_FILES})

foreach(_test_ext ${PYTHON_TEST_FILES})
    set(_test_name pyc_toast_test_${_test_ext})
    add_test(NAME ${_test_name}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND python -c "import toast ; toast.test('${_test_ext}')")    
    set_tests_properties(${_test_name} PROPERTIES LABELS "UnitTest;Python" TIMEOUT 7200)
endforeach(_test_ext ${PYTHON_TEST_FILES})

include(ConfigureSLURM)

