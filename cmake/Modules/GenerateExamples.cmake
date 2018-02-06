
include(MacroUtilities)

#------------------------------------------------------------------------------#

function(machine_options)
    message(STATUS "")
    message(STATUS "Please provide a MACHINE variable from:")
    message(STATUS "\t - cori-haswell")
    message(STATUS "\t - cori-knl")
    message(STATUS "\t - edison")
    message(STATUS "")
endfunction(machine_options)

#------------------------------------------------------------------------------#

function(valid_machine)
    if(NOT "${MACHINE}" STREQUAL "cori-haswell" AND
       NOT "${MACHINE}" STREQUAL "cori-knl" AND
       NOT "${MACHINE}" STREQUAL "edison")
        machine_options()
        message(FATAL_ERROR "invalid MACHINE choice: ${MACHINE}")
    endif()
endfunction(valid_machine)

#------------------------------------------------------------------------------#

function(machine_defined)
    if("${MACHINE}" STREQUAL "" OR NOT DEFINED MACHINE)
        machine_options()
        message(FATAL_ERROR "MACHINE undefined")
    endif("${MACHINE}" STREQUAL "" OR NOT DEFINED MACHINE)
endfunction(machine_defined)

#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# -- Global variable settings
#------------------------------------------------------------------------------#
set(PROBLEM_TYPES ground ground_simple ground_multisite satellite)
set(PROBLEM_SIZES tiny small medium large representative)
set(FREQ_tiny               "Nightly")
set(FREQ_small              "Nightly")
set(FREQ_medium             "Weekly")
set(FREQ_representative     "Weekly")
set(FREQ_large              "Monthly")
set(PROJECT_EXAMPLES_DIR    ${CMAKE_BINARY_DIR}/examples)

#------------------------------------------------------------------------------#

message(STATUS "Cleaning up examples directory...")
execute_process(COMMAND ${PROJECT_EXAMPLES_DIR}/cleanup.sh
    WORKING_DIRECTORY ${PROJECT_EXAMPLES_DIR}
    OUTPUT_FILE cleanup.log
    ERROR_FILE cleanup.log
    RESULT_VARIABLE CLEAN_RET
    TIMEOUT 30)
check_return(CLEAN_RET)
# failed so don't proceed
if(NOT "${CLEAN_RET}" EQUAL 0)
    return()
endif(NOT "${CLEAN_RET}" EQUAL 0)

#------------------------------------------------------------------------------#

message(STATUS "Fetching data for examples...")
execute_process(COMMAND ${PROJECT_EXAMPLES_DIR}/fetch_data.sh
    WORKING_DIRECTORY ${PROJECT_EXAMPLES_DIR}
    OUTPUT_FILE fetch_data.log
    ERROR_FILE fetch_data.log
    RESULT_VARIABLE FETCH_RET
    TIMEOUT 5)
check_return(FETCH_RET)
# failed so don't proceed
if(NOT "${FETCH_RET}" EQUAL 0)
    return()
endif(NOT "${FETCH_RET}" EQUAL 0)

#------------------------------------------------------------------------------#

configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/ctest-wrapper.sh.in
    ${PROJECT_EXAMPLES_DIR}/ctest-wrapper.sh @ONLY)

#------------------------------------------------------------------------------#

message(STATUS "Generating shell examples...")
execute_process(COMMAND ${PROJECT_EXAMPLES_DIR}/generate_shell.sh
    WORKING_DIRECTORY ${PROJECT_EXAMPLES_DIR}
    OUTPUT_FILE generate_shell.log
    ERROR_FILE generate_shell.log
    RESULT_VARIABLE GENERATE_SHELL_RET
    TIMEOUT 30)
check_return(GENERATE_SHELL_RET)
# failed so don't proceed
if(NOT "${GENERATE_SHELL_RET}" EQUAL 0)
    return()
endif(NOT "${GENERATE_SHELL_RET}" EQUAL 0)

#------------------------------------------------------------------------------#
# -- Add shell tests
#------------------------------------------------------------------------------#
foreach(_TYPE ${PROBLEM_TYPES})
    set(_file_name tiny_${_TYPE}_shell)
    set(_test_name pyc_${_file_name}_example)
    # add the test
    add_test(NAME ${_test_name}
        WORKING_DIRECTORY ${PROJECT_EXAMPLES_DIR}
        COMMAND ${PROJECT_EXAMPLES_DIR}/ctest-wrapper.sh
            ${PROJECT_EXAMPLES_DIR}/${_file_name}.sh)
    # set the properties
    set_tests_properties(${_test_name} PROPERTIES 
        LABELS "Examples;shell;tiny;${_TYPE};EXAMPLE_1_NODE" TIMEOUT 7200)
endforeach(_TYPE ground ground_simple satellite)

#------------------------------------------------------------------------------#
# -- return if no SLURM
#------------------------------------------------------------------------------#
if(NOT USE_SLURM OR "${MACHINE}" STREQUAL "")
    return()
endif(NOT USE_SLURM OR "${MACHINE}" STREQUAL "")

# ------------------------------------------------------------------------ #
# -- SLURM examples
# ------------------------------------------------------------------------ #
machine_defined()
valid_machine()

include(${CMAKE_SOURCE_DIR}/examples/templates/machines/${MACHINE})    
set(ENV{MACHINES} "${MACHINE}")

set(PATH_INFO_FILE ctest-path-info.sh)
configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/${PATH_INFO_FILE}.in
    ${PROJECT_EXAMPLES_DIR}/${PATH_INFO_FILE} @ONLY)

FILE(WRITE "${PROJECT_EXAMPLES_DIR}/read.cmake"
"
FILE(READ \"${PROJECT_EXAMPLES_DIR}/${PATH_INFO_FILE}\" _INIT)
set(EXTRA_INIT \"\${_INIT}\" CACHE STRING \"Extra initialization\" FORCE)

")

set(SRUN_WRAPPER srun-wrapper.sh)
configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/${SRUN_WRAPPER}
    ${PROJECT_EXAMPLES_DIR}/${SRUN_WRAPPER} @ONLY)

#------------------------------------------------------------------------------#

message(STATUS "Generating SLURM examples for ${MACHINE}...")
execute_process(COMMAND ${PROJECT_EXAMPLES_DIR}/generate_slurm.sh
        -DACCOUNT=${ACCOUNT} -DTIME=${TIME} -DQUEUE=${QUEUE}
        -DREADFILE=${PROJECT_EXAMPLES_DIR}/read.cmake
    WORKING_DIRECTORY ${PROJECT_EXAMPLES_DIR}
    OUTPUT_FILE generate_slurm.log
    ERROR_FILE generate_slurm.log
    RESULT_VARIABLE GENERATE_SLURM_RET)
check_return(GENERATE_SLURM_RET)

#------------------------------------------------------------------------------#
# SLURM example tests
#------------------------------------------------------------------------------#

foreach(_TYPE ${PROBLEM_TYPES})
    foreach(_SIZE ${PROBLEM_SIZES})
        set(_test_name ${_TYPE}_${_SIZE}_${MACHINE})
        # add the test
        add_test(NAME ${_test_name}
            WORKING_DIRECTORY ${PROJECT_EXAMPLES_DIR}
            COMMAND ${PROJECT_EXAMPLES_DIR}/${SRUN_WRAPPER}
                ${PROJECT_EXAMPLES_DIR}/${_SIZE}_${_TYPE}_${MACHINE}.slurm)
        get_parameter(_NODES
            "${PROJECT_EXAMPLES_DIR}/templates/params/${_TYPE}.${_SIZE}" NODES)
        string(REGEX MATCHALL "([0-9]+)" _NODES "${_NODES}")    
        set(_FREQ "${FREQ_${_SIZE}}")
        string(TOLOWER "${_FREQ}" _LFREQ)
        set(_LABELS Examples ${_TYPE} ${_SIZE} ${_FREQ} Slurm)
        # set the properties
        set_tests_properties(${_test_name} PROPERTIES
            LABELS "${_LABELS}" TIMEOUT 14400)
    endforeach(_SIZE ${PROBLEM_SIZES})
endforeach(_TYPE ${PROBLEM_TYPES})


#------------------------------------------------------------------------------#
