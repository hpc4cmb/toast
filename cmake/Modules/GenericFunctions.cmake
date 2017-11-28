
# - Include guard
if(__genericfunctions_isloaded)
  return()
endif()
set(__genericfunctions_isloaded YES)

include(CMakeParseArguments)

cmake_policy(PUSH)
if(NOT CMAKE_VERSION VERSION_LESS 3.1)
    cmake_policy(SET CMP0054 NEW)
endif()

################################################################################
#   OPTIONS TAKING STRING
################################################################################
function(parse_command_line_options)

    get_cmake_property(CACHE_VARS CACHE_VARIABLES)
    foreach(CACHE_VAR ${CACHE_VARS})

        get_property(CACHE_VAR_HELPSTRING CACHE ${CACHE_VAR} PROPERTY HELPSTRING)
        if(CACHE_VAR_HELPSTRING STREQUAL "No help, variable specified on the command line.")
            get_property(CACHE_VAR_TYPE CACHE ${CACHE_VAR} PROPERTY TYPE)
            if(CACHE_VAR_TYPE STREQUAL "UNINITIALIZED")
                set(CACHE_VAR_TYPE)
            else()
                set(CACHE_VAR_TYPE :${CACHE_VAR_TYPE})
            endif()
            set(CMAKE_ARGS "${CMAKE_ARGS} -D${CACHE_VAR}${CACHE_VAR_TYPE}=\"${${CACHE_VAR}}\"")
        endif()
    endforeach()
    set(CMAKE_ARGS "${CMAKE_ARGS}" PARENT_SCOPE)

endfunction()


################################################################################
#   Macro to add to string
################################################################################
macro(add _VAR _FLAG)
    if(NOT "${_FLAG}" STREQUAL "")
        if("${${_VAR}}" STREQUAL "")
            set(${_VAR} "${_FLAG}")
        else()
            set(${_VAR} "${${_VAR}} ${_FLAG}")
        endif()
    endif()
endmacro()


################################################################################
# function - capitalize - make a string capitalized (first letter is capital)
#   usage:
#       capitalize("SHARED" CShared)
#   message(STATUS "-- CShared is \"${CShared}\"")
#   $ -- CShared is "Shared"
################################################################################
function(capitalize str var)
    # make string lower
    string(TOLOWER "${str}" str)
    string(SUBSTRING "${str}" 0 1 _first)
    string(TOUPPER "${_first}" _first)
    string(SUBSTRING "${str}" 1 -1 _remainder)
    string(CONCAT str "${_first}" "${_remainder}")
    set(${var} "${str}" PARENT_SCOPE)
endfunction()


################################################################################
#   Set project version
################################################################################
macro(SET_PROJECT_VERSION _MAJOR _MINOR _PATCH _DESCRIPTION)
    set(${PROJECT_NAME}_MAJOR_VERSION "${_MAJOR}")
    set(${PROJECT_NAME}_MINOR_VERSION "${_MINOR}")
    set(${PROJECT_NAME}_PATCH_VERSION "${_PATCH}")
    set(${PROJECT_NAME}_VERSION
        "${${PROJECT_NAME}_MAJOR_VERSION}.${${PROJECT_NAME}_MINOR_VERSION}.${${PROJECT_NAME}_PATCH_VERSION}")
    set(${PROJECT_NAME}_SHORT_VERSION
        "${${PROJECT_NAME}_MAJOR_VERSION}.${${PROJECT_NAME}_MINOR_VERSION}")
    set(${PROJECT_NAME}_DESCRIPTION "${_DESCRIPTION}")
    set(PROJECT_VERSION ${${PROJECT_NAME}_VERSION})
endmacro()


################################################################################
#    Get version components
################################################################################
function(GET_VERSION_COMPONENTS OUTPUT_VAR VARIABLE)
    string(REPLACE "." ";" VARIABLE_LIST "${VARIABLE}")
    list(LENGTH VARIABLE_LIST VAR_LENGTH)
    if(VAR_LENGTH GREATER 0)
        list(GET VARIABLE_LIST 0 MAJOR)
        set(${OUTPUT_VAR}_MAJOR_VERSION "${MAJOR}" PARENT_SCOPE)
    endif()
    if(VAR_LENGTH GREATER 1)
        list(GET VARIABLE_LIST 1 MINOR)
        set(${OUTPUT_VAR}_MINOR_VERSION "${MINOR}" PARENT_SCOPE)
        set(${OUTPUT_VAR}_SHORT_VERSION "${MAJOR}.${MINOR}" PARENT_SCOPE)
    endif()
    if(VAR_LENGTH GREATER 2)
        list(GET VARIABLE_LIST 2 PATCH)
        set(${OUTPUT_VAR}_PATCH_VERSION "${PATCH}" PARENT_SCOPE)
    endif()
endfunction()


################################################################################
#    Remove duplicates if string exists
################################################################################
macro(REMOVE_DUPLICATES _ARG)
    if(NOT "${${_ARG}}" STREQUAL "")
        list(REMOVE_DUPLICATES ${_ARG})
    endif()
endmacro()


################################################################################
#   set the two digit version
################################################################################
function(define_version_00 PREFIX POSTFIX)
    if("${PREFIX}" STREQUAL "" OR "${POSTFIX}" STREQUAL "")
        message(WARNING "Unable to define version_00")
        return()
    endif()

    set(_full ${PREFIX}_${POSTFIX}_VERSION)
    if(${_full} LESS 10)
        set(${_full}_00 "0${${_full}}" PARENT_SCOPE)
    else()
        set(${_full}_00 "${${_full}}" PARENT_SCOPE)
    endif()

endfunction()


################################################################################
#   remove a file from a list of files
################################################################################
macro(REMOVE_FILE _LIST _NAME)

    # loop over the list
    foreach(_FILE ${${_LIST}})
        # get the base filename
        get_filename_component(_BASE ${_FILE} NAME)
        # check if full filename or base filename is exact match for
        # name provided
        if("${_BASE}" STREQUAL "${_NAME}" OR
           "${_FILE}" STREQUAL "${_NAME}")
            # remove from list
            list(REMOVE_ITEM ${_LIST} ${_FILE})

        endif("${_BASE}" STREQUAL "${_NAME}" OR
              "${_FILE}" STREQUAL "${_NAME}")

    endforeach(_FILE ${${_LIST}})

endmacro(REMOVE_FILE _LIST _NAME)



cmake_policy(POP)
