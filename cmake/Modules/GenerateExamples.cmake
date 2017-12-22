
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
set(FREQ_tiny           "Nightly")
set(FREQ_small          "Nightly")
set(FREQ_medium         "Weekly")
set(FREQ_representative "Weekly")
set(FREQ_large          "Monthly")

#------------------------------------------------------------------------------#

message(STATUS "Cleaning up examples directory...")
execute_process(COMMAND ./cleanup.sh
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    OUTPUT_FILE cleanup.log
    ERROR_FILE cleanup.log
    RESULT_VARIABLE CLEAN_RET)
check_return(CLEAN_RET)

#------------------------------------------------------------------------------#

message(STATUS "Fetching data for examples...")
execute_process(COMMAND ./fetch_data.sh
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    OUTPUT_FILE fetch_data.log
    ERROR_FILE fetch_data.log
    RESULT_VARIABLE FETCH_RET)
check_return(FETCH_RET)

#------------------------------------------------------------------------------#

configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/ctest-wrapper.sh.in
    ${CMAKE_BINARY_DIR}/examples/ctest-wrapper.sh @ONLY)

#------------------------------------------------------------------------------#

message(STATUS "Generating shell examples...")
execute_process(COMMAND ./generate_shell.sh
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    OUTPUT_FILE generate_shell.log
    ERROR_FILE generate_shell.log
    RESULT_VARIABLE GENERATE_SHELL_RET)
check_return(GENERATE_SHELL_RET)

#------------------------------------------------------------------------------#
# -- Add shell tests
#------------------------------------------------------------------------------#
foreach(_TYPE ${PROBLEM_TYPES})
    set(_file_name tiny_${_TYPE}_shell)
    set(_test_name pyc_${_file_name}_example)
    add_test(NAME ${_test_name}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    COMMAND ${SLURM_COMMAND} ./ctest-wrapper.sh ${CMAKE_BINARY_DIR}/examples/${_file_name}.sh)
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
    ${CMAKE_BINARY_DIR}/examples/${PATH_INFO_FILE} @ONLY)

FILE(WRITE "${CMAKE_BINARY_DIR}/examples/read.cmake"
"
FILE(READ \"${CMAKE_BINARY_DIR}/examples/${PATH_INFO_FILE}\" _INIT)
set(EXTRA_INIT \"\${_INIT}\" CACHE STRING \"Extra initialization\" FORCE)

")

set(SRUN_WRAPPER srun-wrapper.sh)
configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/${SRUN_WRAPPER}
    ${CMAKE_BINARY_DIR}/examples/${SRUN_WRAPPER} @ONLY)

#------------------------------------------------------------------------------#

message(STATUS "Generating SLURM examples for ${MACHINE}...")
execute_process(COMMAND ./generate_slurm.sh -DACCOUNT=${ACCOUNT} -DTIME=${TIME} 
        -DQUEUE=${QUEUE} -DREADFILE=${CMAKE_BINARY_DIR}/examples/read.cmake
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    OUTPUT_FILE generate_slurm.log
    ERROR_FILE generate_slurm.log
    RESULT_VARIABLE GENERATE_SLURM_RET)
check_return(GENERATE_SLURM_RET)

#------------------------------------------------------------------------------#

foreach(_TYPE ${PROBLEM_TYPES})
    foreach(_SIZE ${PROBLEM_SIZES})
        set(_test_name ${_TYPE}_${_SIZE}_${MACHINE})
        add_test(NAME ${_test_name}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples
            COMMAND ./${SRUN_WRAPPER} ${CMAKE_BINARY_DIR}/examples/${_SIZE}_${_TYPE}_${MACHINE}.slurm)
        get_parameter(_NODES "${CMAKE_BINARY_DIR}/examples/templates/params/${_TYPE}.${_SIZE}" NODES)
        string(REGEX MATCHALL "([0-9]+)" _NODES "${_NODES}")    
        set(_FREQ "${FREQ_${_SIZE}}")
        string(TOLOWER "${_FREQ}" _LFREQ)
        set(_LABELS  Examples${_TYPE} ${MACHINE} ${_SIZE} ${_FREQ}
            ${_TYPE}_${_SIZE} ${_TYPE}_${_LFREQ}
            ${_TYPE}_${_NODES} ${_LFREQ}_${_NODES}
            EXAMPLE_${_NODES}_NODE "slurm")
            list(APPEND NODES_LIST ${_NODES})
            if(NOT DEFINED NODES_COUNT_${_NODES})
                set(NODES_COUNT_${_NODES} 0)
            endif(NOT DEFINED NODES_COUNT_${_NODES})
            math(EXPR NODES_COUNT_${_NODES} "${NODES_COUNT_${_NODES}}+1")
        set_tests_properties(${_test_name} PROPERTIES LABELS "${_LABELS}" TIMEOUT 14400)
    endforeach(_SIZE ${PROBLEM_SIZES})
endforeach(_TYPE ${PROBLEM_TYPES})
   
#------------------------------------------------------------------------------#

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

#------------------------------------------------------------------------------#
