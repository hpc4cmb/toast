
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

function(check_return _VAR)
    if("${${_VAR}}" GREATER 0)
        message(WARNING "Error code for ${_VAR} is greater than zero: ${${_VAR}}")
    endif("${${_VAR}}" GREATER 0)
endfunction(check_return _VAR)

#------------------------------------------------------------------------------#

enum_option(MACHINE
    DOC "Machine to run CTest SLURM scripts"
    VALUES cori-knl cori-haswell edison
    CASE_INSENSITIVE
)   
add_feature(MACHINE "SLURM machine")

machine_defined()
valid_machine()
    
message(STATUS "Cleaning up examples directory...")
execute_process(COMMAND ./cleanup.sh
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    OUTPUT_FILE cleanup.log
    ERROR_FILE cleanup.log
    RESULT_VARIABLE CLEAN_RET)
check_return(CLEAN_RET)

message(STATUS "Fetching data for examples...")
execute_process(COMMAND ./fetch_data.sh
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    OUTPUT_FILE fetch_data.log
    ERROR_FILE fetch_data.log
    RESULT_VARIABLE FETCH_RET)
check_return(FETCH_RET)

include(${CMAKE_SOURCE_DIR}/examples/templates/machines/${MACHINE})
    
set(ENV{MACHINES} "${MACHINE}")

message(STATUS "Generating SLURM examples for ${MACHINE}...")
execute_process(COMMAND ./generate_slurm.sh -DACCOUNT=${ACCOUNT} -DTIME=${TIME} -DQUEUE=${QUEUE}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    OUTPUT_FILE generate_slurm.log
    ERROR_FILE generate_slurm.log
    RESULT_VARIABLE GENERATE_RET)
check_return(GENERATE_RET)

set(PROBLEM_TYPES ground ground_simple satellite)
set(PROBLEM_SIZES tiny small medium large representative)

foreach(_TYPE ${PROBLEM_TYPES})
    foreach(_SIZE ${PROBLEM_SIZES})
        set(_test_name slurm_${_TYPE}_${_SIZE}_${MACHINE})
        set(JOB_NAME "${_TYPE}_${_SIZE}_${MACHINE}")
        set(CTEST_SCRIPT_NAME ${_SIZE}_${_TYPE}_${MACHINE}.slurm)
        add_test(NAME ${_test_name}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples
            COMMAND salloc ${CTEST_SCRIPT_NAME})
        set_tests_properties(${_test_name} PROPERTIES LABELS "Examples;${_TYPE};${MACHINE};${_SIZE};Nightly" TIMEOUT 14400)
    endforeach(_SIZE ${PROBLEM_SIZES})
endforeach(_TYPE ${PROBLEM_TYPES})
   
    
