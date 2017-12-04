
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
# -- Configure CTest
# ------------------------------------------------------------------------ #

GET_TEMPORARY_DIRECTORY(CMAKE_DASHBOARD_ROOT "${CMAKE_PROJECT_NAME}-cdash")

## -- USE_<PROJECT> and <PROJECT>_ROOT
set(CMAKE_CONFIGURE_OPTIONS )
foreach(_OPTION BLAS FFTW LAPACK MKL MPI OPENMP PYTHON SSE TBB WCSLIB CLANG_ASSEMBLER
        ELEMENTAL MATH)
    add(CMAKE_CONFIGURE_OPTIONS "-DUSE_${_OPTION}=${USE_${_OPTION}}")
    if(NOT "${${_OPTION}_ROOT}" STREQUAL "")
        add(CMAKE_CONFIGURE_OPTIONS "-D${_OPTION}_ROOT=${${_OPTION}_ROOT}")
    endif(NOT "${${_OPTION}_ROOT}" STREQUAL "")
endforeach()

execute_process(COMMAND git branch --contains
    OUTPUT_VARIABLE CMAKE_SOURCE_BRANCH
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "*" "" CMAKE_SOURCE_BRANCH "${CMAKE_SOURCE_BRANCH}")
string(REPLACE " " "" CMAKE_SOURCE_BRANCH "${CMAKE_SOURCE_BRANCH}")

## -- CTest Config
configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/CTestConfig.cmake.in
    ${CMAKE_BINARY_DIR}/CTestConfig.cmake @ONLY)

ENABLE_TESTING()
include(CTest)

## -- CTest Setup
configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/CTestSetup.cmake.in
    ${CMAKE_BINARY_DIR}/CTestSetup.cmake @ONLY)

## -- CTest Custom
configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/CTestCustom.cmake.in
    ${CMAKE_BINARY_DIR}/CTestCustom.cmake @ONLY)

add_test(NAME toast_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_BINARY_DIR}/toast_test)
