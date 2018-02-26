
ENABLE_TESTING()
if(BUILD_TESTING)
    include(CTest)
endif(BUILD_TESTING)

# if this is directory we are running CDash (don't set to ON)
add_option(DASHBOARD_MODE
    "Internally used to skip generation of CDash files" OFF NO_FEATURE)
# slurm config
add_dependent_option(USE_SLURM "Enable generation of SLURM scripts for testing"
    ON "SLURM_SBATCH_COMMAND" OFF)
    
set(TIME "00:30:00" CACHE STRING "SLURM runtime")
set(ACCOUNT "dasrepo" CACHE STRING "SLURM account")
set(QUEUE "regular" CACHE STRING "SLURM queue")
set_ifnot(JOB_NAME "UnitTest")
set_ifnot(NODES 1)

function(convert_time VAR VAL)
    string(REPLACE ":" ";" VAL "${VAL}")
    list(GET VAL 0 HOUR)
    list(GET VAL 1 MIN)
    math(EXPR _TIME "${HOUR}*60+${MIN}")
    set(${VAR} ${_TIME} PARENT_SCOPE)
endfunction(convert_time VAR VAL)

# ------------------------------------------------------------------------ #
# -- SLURM variables
# ------------------------------------------------------------------------ #
if(USE_SLURM)
    # first value is default so if TARGET_ARCHITECTURE is known, then use it
    set(_MACHINE_ORDERING cori-knl cori-haswell edison)
    if("${TARGET_ARCHITECTURE}" STREQUAL "haswell")
        set(_MACHINE_ORDERING cori-haswell cori-knl edison)
    endif("${TARGET_ARCHITECTURE}" STREQUAL "haswell")
    # set the machines (first value is default)
    enum_option(MACHINE VALUES ${_MACHINE_ORDERING}
        DOC "Machine to run CTest SLURM scripts" CASE_INSENSITIVE)
    add_feature(ACCOUNT "SLURM account")
    add_feature(QUEUE "SLURM queue")
    add_feature(MACHINE "SLURM machine")
    STRING(REPLACE "cori-" "" _MACHINE "${MACHINE}")
    set(MACHINEFILE "${CMAKE_BINARY_DIR}/examples/templates/machines/${MACHINE}")
    get_parameter(CONSTRAINT ${MACHINEFILE} CONSTRAINT)
    foreach(VAR NODEPROCS NODECORES HYPERTHREAD)
        get_parameter(${VAR} ${MACHINEFILE} ${VAR})
        string(REGEX MATCHALL "([0-9]+)" ${VAR} "${${VAR}}")
    endforeach(VAR NODEPROCS NODECORES HYPERTHREAD)
    math(EXPR NODEDEPTH "${NODECORES}/${NODEPROCS}")
endif(USE_SLURM)


# ------------------------------------------------------------------------ #
# -- Miscellaneous
# ------------------------------------------------------------------------ #
if(NOT DASHBOARD_MODE AND BUILD_TESTING)
    add_option(CTEST_LOCAL_CHECKOUT "Use the local source tree for CTest/CDash" OFF)
    if(CTEST_LOCAL_CHECKOUT)
        set_ifnot(CMAKE_LOCAL_DIRECTORY "${CMAKE_SOURCE_DIR}")
        set(CTEST_MODEL "Continuous" CACHE STRING "Model for CTest")
    else(CTEST_LOCAL_CHECKOUT)
        set(CTEST_MODEL "Nightly" CACHE STRING "Model for CTest")
    endif(CTEST_LOCAL_CHECKOUT)
    add_feature(CTEST_MODEL "Model for CDash")
endif(NOT DASHBOARD_MODE AND BUILD_TESTING)

set(ENV{CTEST_USE_LAUNCHERS_DEFAULT} ${CMAKE_BINARY_DIR}/CDashGlob.cmake)
# environment variables
foreach(_ENV PATH LD_LIBRARY_PATH DYLD_LIBRARY_PATH PYTHONPATH)
    set(${_ENV} "$ENV{${_ENV}}")
endforeach(_ENV PATH LD_LIBRARY_PATH DYLD_LIBRARY_PATH PYTHONPATH)

configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/ctest-wrapper.sh.in
    ${CMAKE_BINARY_DIR}/ctest-wrapper.sh @ONLY)


# ------------------------------------------------------------------------ #
# -- Function to create a temporary directory
# ------------------------------------------------------------------------ #
function(GET_TEMPORARY_DIRECTORY DIR_VAR DIR_BASE DIR_MODEL)
    # create a root working directory
    set(_TMP_ROOT "${CMAKE_BINARY_DIR}/cdash/${DIR_MODEL}")
    set(${DIR_VAR} "${_TMP_ROOT}" PARENT_SCOPE)
    return()
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${_TMP_ROOT}) 
    
    # make temporary directory
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
        execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${_TMP_STR})
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
string(REPLACE "* " "" CMAKE_SOURCE_BRANCH "${CMAKE_SOURCE_BRANCH}")
string(REPLACE "*" "" CMAKE_SOURCE_BRANCH "${CMAKE_SOURCE_BRANCH}")
# removes things such as : '(HEAD detached at 3109eab) master' -> 'master'
string(REGEX REPLACE "(\\\(.*\\\) )" "" CMAKE_SOURCE_BRANCH "${CMAKE_SOURCE_BRANCH}")


# ------------------------------------------------------------------------ #
# -- Add options
# ------------------------------------------------------------------------ #
macro(add_ctest_options VARIABLE )
    get_cmake_property(_vars CACHE_VARIABLES)
    get_cmake_property(_nvars VARIABLES)
    foreach(_var ${_nvars})
        list(APPEND _vars ${_var})
    endforeach(_var ${_nvars})

    list(REMOVE_DUPLICATES _vars)
    list(SORT _vars)
    set(_set_vars ${ARGN})
    foreach(_var ${_vars})
        STRING(REGEX MATCH "^USE_" _use_found "${_var}")
        STRING(REGEX MATCH ".*(_ROOT|_LIBRARY|_INCLUDE_DIR|_INCLUDE_PATH|_EXECUTABLE)$"
            _root_found "${_var}")
        STRING(REGEX MATCH "^(PREVIOUS_|CMAKE_|OSX_|DEFAULT_|EXTERNAL_|_)"
            _skip_prefix "${_var}")
        STRING(REGEX MATCH ".*(_AVAILABLE|_LIBRARIES|_INCLUDE_DIRS)$"
            _skip_suffix "${_var}")

        if(_skip_prefix OR _skip_suffix)
            continue()
        endif(_skip_prefix OR _skip_suffix)

        if(_use_found OR _root_found)
            list(APPEND _set_vars ${_var})
        endif(_use_found OR _root_found)
    endforeach(_var ${_vars})

    list(REMOVE_DUPLICATES _set_vars)
    list(SORT _set_vars)
    foreach(_var ${_set_vars})
        if(NOT "${${_var}}" STREQUAL "" AND
            NOT "${${_var}}" STREQUAL "${_var}-NOTFOUND")
            add(${VARIABLE} "-D${_var}='${${_var}}'")
        endif(NOT "${${_var}}" STREQUAL "" AND
            NOT "${${_var}}" STREQUAL "${_var}-NOTFOUND")
    endforeach(_var ${_set_vars})

    #string(REPLACE " " "\n\t" _tmp "${${VARIABLE}}")
    #message(STATUS "${VARIABLE} : \n\t${_tmp}")
    unset(_vars)
    unset(_nvars)
    unset(_set_vars)

endmacro(add_ctest_options VARIABLE )


# ------------------------------------------------------------------------ #
# -- Configure CTest for CDash
# ------------------------------------------------------------------------ #
if(NOT DASHBOARD_MODE AND BUILD_TESTING)
    # get temporary directory for dashboard testing
    if(NOT DEFINED CMAKE_DASHBOARD_ROOT)
        GET_TEMPORARY_DIRECTORY(CMAKE_DASHBOARD_ROOT
            "${CMAKE_PROJECT_NAME}-cdash" ${CTEST_MODEL})
    endif(NOT DEFINED CMAKE_DASHBOARD_ROOT)
    add_feature(CMAKE_DASHBOARD_ROOT "Dashboard directory")
    
    # set the CMake configure options
    add_ctest_options(CMAKE_CONFIGURE_OPTIONS
        ACCOUNT MACHINE QUEUE
        TARGET_ARCHITECTURE
        CTEST_LOCAL_CHECKOUT CMAKE_BUILD_TYPE
        CMAKE_C_COMPILER CMAKE_CXX_COMPILER
        MPI_C_COMPILER MPI_CXX_COMPILER
        CTEST_MODEL CTEST_SITE GENERATE_EXAMPLES 
        PYTHON_EXECUTABLE)

    ## -- CTest Config
    configure_file(${CMAKE_SOURCE_DIR}/CTestConfig.cmake
        ${CMAKE_BINARY_DIR}/CTestConfig.cmake @ONLY)
    
    set(cdash_templates Init Build Test Submit Glob Stages)
    if(USE_COVERAGE)
        list(APPEND cdash_templates Coverage)
    endif(USE_COVERAGE)
    if(MEMORYCHECK_COMMAND)
        list(APPEND cdash_templates MemCheck)
    endif(MEMORYCHECK_COMMAND)
    foreach(_type ${cdash_templates})
        ## -- CTest Setup
        configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/CDash${_type}.cmake.in
            ${CMAKE_BINARY_DIR}/cdash/${_type}.cmake @ONLY)
    endforeach(_type Init Build Test Coverage MemCheck Submit Glob Stages)

    ## -- CTest Custom
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/CTestCustom.cmake.in
        ${CMAKE_BINARY_DIR}/CTestCustom.cmake @ONLY)

endif(NOT DASHBOARD_MODE AND BUILD_TESTING)

# ------------------------------------------------------------------------ #
# -- Configure CTest tests
# ------------------------------------------------------------------------ #
set(SLURM_COMMAND )
if(USE_SLURM)
    set(SLURM_COMMAND ${SLURM_SRUN_COMMAND} -n 1)
endif(USE_SLURM)

# ------------------------------------------------------------------------ #
# get the python tests
set(_PYDIR "${CMAKE_SOURCE_DIR}/src/python/tests")
FILE(GLOB _PYTHON_TEST_FILES "${_PYDIR}/*.py")
LIST(REMOVE_ITEM _PYTHON_TEST_FILES ${_PYDIR}/__init__.py ${_PYDIR}/runner.py)
set(PYTHON_TEST_FILES )
foreach(_FILE ${_PYTHON_TEST_FILES})
    get_filename_component(_FILE_NE "${_FILE}" NAME_WE)
    list(APPEND PYTHON_TEST_FILES "${_FILE_NE}")
endforeach(_FILE ${_PYTHON_TEST_FILES})

# ------------------------------------------------------------------------ #
# add CXX unit test
add_test(NAME cxx_toast_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_BINARY_DIR}/ctest-wrapper.sh
        ${SLURM_COMMAND} ${CMAKE_BINARY_DIR}/toast_test)
# set the properties
set_tests_properties(cxx_toast_test PROPERTIES
    LABELS "UnitTest;CXX" TIMEOUT 7200)

# ------------------------------------------------------------------------ #
# add Python unit tests
foreach(_test_ext ${PYTHON_TEST_FILES})
    # name of python test
    set(_test_name pyc_toast_test_${_test_ext})
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/pytestscript.sh.in
        ${CMAKE_BINARY_DIR}/scripts/${_test_name}.sh @ONLY)
    # add the test
    add_test(NAME ${_test_name}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND ${SLURM_COMMAND} ${CMAKE_BINARY_DIR}/scripts/${_test_name}.sh)
    # set the properties
    set_tests_properties(${_test_name} PROPERTIES
        LABELS "UnitTest;Python" TIMEOUT 7200)
endforeach(_test_ext ${PYTHON_TEST_FILES})

# ------------------------------------------------------------------------ #
# coverage
if(USE_COVERAGE)
    # add the test
    add_test(NAME pyc_coverage
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND ${SLURM_COMMAND} ${CMAKE_BINARY_DIR}/ctest-wrapper.sh
            ${PYTHON_EXECUTABLE}
            ${CMAKE_BINARY_DIR}/pyc_toast_test_coverage.py)
    # set the properties
    set_tests_properties(pyc_coverage PROPERTIES 
        LABELS "Coverage;Python" TIMEOUT 7200)
endif(USE_COVERAGE)


# ------------------------------------------------------------------------ #
# examples
add_option(GENERATE_EXAMPLES "Generate TOAST examples for testing" ON)
if(GENERATE_EXAMPLES)
    include(GenerateExamples)
endif(GENERATE_EXAMPLES)
