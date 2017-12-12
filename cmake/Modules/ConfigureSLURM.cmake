
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

machine_defined()
valid_machine()

set(ENV{GENERATE_SHELL} 1)

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

message(STATUS "Generating examples for ${MACHINE}...")
execute_process(COMMAND ./generate_shell.sh -DACCOUNT=${ACCOUNT} -DTIME=${TIME} -DQUEUE=${QUEUE}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    OUTPUT_FILE generate_shell.log
    ERROR_FILE generate_shell.log
    RESULT_VARIABLE GENERATE_SHELL_RET)
check_return(GENERATE_SHELL_RET)

execute_process(COMMAND ./generate_slurm.sh -DACCOUNT=${ACCOUNT} -DTIME=${TIME} -DQUEUE=${QUEUE}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    OUTPUT_FILE generate_slurm.log
    ERROR_FILE generate_slurm.log
    RESULT_VARIABLE GENERATE_SLURM_RET)
check_return(GENERATE_SLURM_RET)

set(PROBLEM_TYPES ground ground_simple satellite)
set(PROBLEM_SIZES tiny small medium large representative)

set(FREQ_tiny           "Nightly")
set(FREQ_small          "Nightly")
set(FREQ_medium         "Weekly")
set(FREQ_representative "Weekly")
set(FREQ_large          "Monthly")

foreach(_TYPE ${PROBLEM_TYPES})
    foreach(_SIZE ${PROBLEM_SIZES})
        set(_test_name ${_TYPE}_${_SIZE}_${MACHINE})
        add_test(NAME ${_test_name}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples
            COMMAND ctest-wrapper.sh ${_SIZE}_${_TYPE}_${MACHINE}.slurm)
        get_parameter(_NODES "${CMAKE_BINARY_DIR}/examples/templates/params/${_TYPE}.${_SIZE}" NODES)
        string(REGEX MATCHALL "([0-9]+)" _NODES "${_NODES}")    
        set(_FREQ "${FREQ_${_SIZE}}")
        string(TOLOWER "${_FREQ}" _LFREQ)
        set(_LABELS 
            Examples 
            ${_TYPE} 
            ${MACHINE} 
            ${_SIZE} 
            ${_FREQ} 
            ${_TYPE}_${_SIZE} 
            ${_TYPE}_${_LFREQ}
            EXAMPLE_${_NODES}_NODE 
            ${_TYPE}_${_NODES}
            ${_LFREQ}_${_NODES}
            )
            list(APPEND NODES_LIST ${_NODES})
            math(EXPR NODES_COUNT_${_NODES} "${NODES_COUNT_${_NODES}}+1")
        set_tests_properties(${_test_name} PROPERTIES LABELS "${_LABELS}" TIMEOUT 14400)
    endforeach(_SIZE ${PROBLEM_SIZES})
endforeach(_TYPE ${PROBLEM_TYPES})
   
list(REMOVE_DUPLICATES NODES_LIST)
numeric_sort(NODES_LIST ${NODES_LIST})
set(_STR )
foreach(_N ${NODES_LIST})
    set(_C "${NODES_COUNT_${_N}}")
    set(_STR "${_STR}${_N} ${NODES_COUNT_${_N}}\n")
    math(EXPR _MIN_TOT "30*${_C}+45")
    math(EXPR _MIN_REM "${_MIN_TOT}%60")
    math(EXPR _HRS_TOT "(${_MIN_TOT}-${_MIN_REM})/60")
    get_00(_MIN_REM "${_MIN_REM}")
    get_00(_HRS_TOT "${_HRS_TOT}")
    set(NODES ${_N})
    set(TIME "${_HRS_TOT}:${_MIN_REM}:00")
    set(JOB_NAME "example_${_N}_node")
    set(FILENAME "example_${_N}_node.sh")
    set(CTEST_PARAMS "-L EXAMPLE_${_N}_NODE -S CDashTest.cmake -O ctest_${JOB_NAME}.log --output-on-failure")
    set(QUEUE_ORIG "${QUEUE}")
    set(QUEUE "debug")
    if(_MIN_TOT GREATER 120)
        set(QUEUE "regular")
    endif(_MIN_TOT GREATER 120)
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/CTestExample.sh.in
        ${CMAKE_BINARY_DIR}/${FILENAME} @ONLY)
    set(QUEUE "${QUEUE_ORIG}")
endforeach(_N ${NODES_LIST})
file(WRITE ${CMAKE_BINARY_DIR}/examples/nodes.cmake.out "${_STR}")

