
include(CMakeParseArguments)

#==============================================================================#

cmake_policy(PUSH)
if(NOT CMAKE_VERSION VERSION_LESS 3.1)
    cmake_policy(SET CMP0054 NEW)
endif()

find_program(GIT_COMMAND git)

################################################################################
#
#   Write a config file for installation (writes preprocessor definitions)
#
################################################################################

function(write_toast_config_file)

    cmake_parse_arguments(CONF
        ""
        "TEMPLATE;FILE_NAME;WRITE_PATH;INSTALL_PATH"
        "DEFINE_VARS;UNDEFINE_VARS;SKIP_VARS"
        ${ARGN}
    )


    get_directory_property(_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)

    if(CONF_TEMPLATE)
        configure_file(${CONF_TEMPLATE} ${CMAKE_BINARY_DIR}/.${CONF_FILE_NAME}.tmp @ONLY)
        file(READ ${CMAKE_BINARY_DIR}/.${CONF_FILE_NAME}.tmp _orig)
    endif()

    STRING(TOLOWER ${PROJECT_NAME} _project)
    STRING(TOLOWER ${CONF_FILE_NAME} _file)
    set(_name "${_project}_${_file}")
    STRING(REPLACE "." "_" _name "${_name}")
    STRING(REPLACE "/" "_" _name "${_name}")
    STRING(TOLOWER "${_name}" _name)

    set(header1 "#ifndef ${_name}_")
    set(header2 "#define ${_name}_")

    set(_output "${_orig}//\n//\n//\n//\n//\n\n${header1}\n${header2}\n")

    # write the version fields
    if(DEFINED ${PROJECT_NAME}_VERSION)
        define_version_00(${PROJECT_NAME} MAJOR)
        define_version_00(${PROJECT_NAME} MINOR)
        define_version_00(${PROJECT_NAME} PATCH)

        set(_prefix ${PROJECT_NAME})
        if("${${_prefix}_PATCH}" GREATER 0)
            set(VERSION_STRING "${${_prefix}_MAJOR_VERSION}")
            set(VERSION_STRING "${VERSION_STRING}_${${_prefix}_MINOR_VERSION}")
            set(VERSION_STRING "${VERSION_STRING}_${${_prefix}_PATCH_VERSION}")
        else()
            set(VERSION_STRING "${${_prefix}_MAJOR_VERSION}_${${_prefix}_MINOR_VERSION}")
        endif()

        set(s1 "//")
        set(s2 "//  Caution, this is the only ${PROJECT_NAME} header that is guaranteed")
        set(s3 "//  to change with every release, including this header")
        set(s4 "//  will cause a recompile every time a new ${PROJECT_NAME} version is")
        set(s5 "//  released.")
        set(s6 "//")
        set(s7 "//  ${PROJECT_NAME}_VERSION % 100 is the patch level")
        set(s8 "//  ${PROJECT_NAME}_VERSION / 100 % 1000 is the minor version")
        set(s9 "//  ${PROJECT_NAME}_VERSION / 100000 is the major version")

        set(_output "${_output}\n${s1}\n${s2}\n${s3}\n${s4}\n${s5}\n${s6}\n${s7}\n${s8}\n${s9}\n")
        set(_output "${_output}\n#define ${PROJECT_NAME}_VERSION")
        set(_output "${_output} ${${_prefix}_MAJOR_VERSION_00}")
        set(_output "${_output}${${_prefix}_MINOR_VERSION_00}")
        set(_output "${_output}${${_prefix}_PATCH_VERSION_00}\n")

        set(s1 "//")
        set(s2 "//  ${PROJECT_NAME}_LIB_VERSION must be defined to be the same as")
        set(s3 "//  ${PROJECT_NAME}_VERSION but as a *string* in the form \"x_y[_z]\" where x is")
        set(s4 "//  the major version number, y is the minor version number, and z is the patch")
        set(s5 "//  level if not 0.")

        set(_output "${_output}\n${s1}\n${s2}\n${s3}\n${s4}\n${s5}\n")
        set(_output "${_output}\n#define ${PROJECT_NAME}_LIB_VERSION \"${VERSION_STRING}\"\n")
    endif()

    set(_output "${_output}\n//\n// TOAST CONFIG\n//\n")

    foreach(_def ${CONF_DEFINE_VARS})
        set(_str0 "/* Define ${${_def}_LABEL} ${${_def}_MSG} */")
        set(_str1 "#define ${${_def}_MACRO} ${${_def}_ENTRY}")
        set(_output "${_output}\n${_str0}\n${_str1}\n")
    endforeach()

    foreach(_def ${CONF_UNDEFINE_VARS})
        set(_str0 "/* Define ${${_def}_LABEL} ${${_def}_MSG} */")
        set(_str1 "/* #undef ${${_def}_LABEL} */")
        set(_output "${_output}\n${_str0}\n${_str1}\n")
    endforeach()

    set(_output "${_output}\n//\n// DEFINES\n//\n")

    set(_FUTURE_REMOVE DEBUG NDEBUG ${CONF_SKIP_VARS})
    # if DEBUG, add ${PROJECT_NAME}_DEBUG and same for NDEBUG
    foreach(_def ${_defs})
        if("${_def}" STREQUAL "DEBUG")
            list(APPEND _defs ${PROJECT_NAME}_DEBUG)
        elseif("${_def}" STREQUAL "NDEBUG")
            list(APPEND _defs ${PROJECT_NAME}_NDEBUG)
        elseif("${_def}" MATCHES "=")
            list(APPEND _FUTURE_REMOVE ${_def})
        endif()
    endforeach()

    foreach(_def ${_FUTURE_REMOVE})
        list(REMOVE_ITEM _defs ${_def})
    endforeach()
    unset(_FUTURE_REMOVE)

    foreach(_def ${_defs})
        set(_str1 "#if !defined(${_def})")
        set(_str2 "#   define ${_def}")
        set(_str3 "#endif")
        set(_output "${_output}\n${_str1}\n${_str2}\n${_str3}\n")
    endforeach()

    list(APPEND _defs ${CMAKE_PROJECT_NAME}_DEBUG)
    list(APPEND _defs ${CMAKE_PROJECT_NAME}_NDEBUG)
    list(REMOVE_DUPLICATES _defs)

    set(_output "${_output}\n//\n// To avoid any of the definitions, define DONT_{definition}\n//\n")
    foreach(_def ${_defs})
        set(_str1 "#if defined(DONT_${_def})")
        set(_str2 "#   if defined(${_def})")
        set(_str3 "#       undef ${_def}")
        set(_str4 "#   endif")
        set(_str5 "#endif")
        set(_output "${_output}\n${_str1}\n${_str2}\n${_str3}\n${_str4}\n${_str5}\n")
    endforeach()

    set(_output "${_output}\n\n#endif // end ${_name}_\n\n")

    get_filename_component(_fname ${CONF_FILE_NAME} NAME)
    if(NOT EXISTS ${CONF_WRITE_PATH})
        execute_process(COMMAND
            ${CMAKE_COMMAND} -E make_directory ${CONF_WRITE_PATH})
    endif()
    if(NOT EXISTS ${CONF_WRITE_PATH}/${_fname})
        file(WRITE ${CONF_WRITE_PATH}/${_fname} ${_output})
    else()
        file(WRITE ${CMAKE_BINARY_DIR}/.config_diff/${_fname} ${_output})
        file(STRINGS ${CONF_WRITE_PATH}/${_fname} _existing)
        file(STRINGS ${CMAKE_BINARY_DIR}/.config_diff/${_fname} _just_written)
        string(COMPARE EQUAL "${_existing}" "${_just_written}" _diff)
        if(NOT _diff)
            file(WRITE ${CONF_WRITE_PATH}/${_fname} ${_output})
        endif()
    endif()

    if(NOT "${CONF_INSTALL_PATH}" STREQUAL "")
        install(FILES ${CONF_WRITE_PATH}/${_fname} DESTINATION ${CONF_INSTALL_PATH})
    endif()

endfunction()

#==============================================================================#
#   Function to convert a boolean ON/OFF into DEFINE/UNDEFINE
#       flag for SET_MACRO_FIELDS (below)
#
function(GET_DEFINED_FLAG VAR QUERY)

    if(${QUERY})
        SET(${VAR} DEFINE PARENT_SCOPE)
    else()
        SET(${VAR} UNDEFINE PARENT_SCOPE)
    endif(${QUERY})

endfunction(GET_DEFINED_FLAG VAR QUERY)

#==============================================================================#
#   function to assign the corresponding macro fields for LABEL "X"
#   - e.g. label "HAVE_XML" needs to define:
#       HAVE_XML_LABEL  - the name of the macro
#       HAVE_XML_MACRO   - the signature of macro (only different from
#                         LABEL if macro is function i.e., F77_FUNC(xxx)
#       HAVE_XML_ENTRY  - what to set define statement to
#       HAVE_XML_MSG    - the comment
#
function(SET_MACRO_FIELDS)
    cmake_parse_arguments(
        VAR
        "DEFINE;UNDEFINE"
        "MACRO;ENTRY;LABEL;MSG"
        ""
        ${ARGN})

    # LABEL and MACRO are *typically* the same
    if("${VAR_MACRO}" STREQUAL "" AND NOT "${VAR_LABEL}" STREQUAL "")
        set(VAR_MACRO "${VAR_LABEL}")
    elseif(NOT "${VAR_MACRO}" STREQUAL "" AND "${VAR_LABEL}" STREQUAL "")
        set(VAR_LABEL "${VAR_MACRO}")
    endif()

    set(${VAR_LABEL}_MACRO  "${VAR_MACRO}"  PARENT_SCOPE)
    set(${VAR_LABEL}_ENTRY  "${VAR_ENTRY}"  PARENT_SCOPE)
    set(${VAR_LABEL}_LABEL  "${VAR_LABEL}"  PARENT_SCOPE)
    set(${VAR_LABEL}_MSG    "${VAR_MSG}"    PARENT_SCOPE)

    if(VAR_DEFINE)
        set(DEFINE_VARIABLES ${DEFINE_VARIABLES} ${VAR_LABEL} PARENT_SCOPE)
    elseif(VAR_UNDEFINE)
        set(UNDEFINE_VARIABLES ${UNDEFINE_VARIABLES} ${VAR_LABEL} PARENT_SCOPE)
    endif()

endfunction(SET_MACRO_FIELDS)

#==============================================================================#
#   function to see if header exists
#   - it creates an empty main that includes the header and checks that
#     it compiles
#
function(CHECK_FOR_HEADER VAR HEADER)

    get_property(dirs DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

    set(HEADER_FILE ${HEADER})
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/header-test.cc.in
        ${CMAKE_BINARY_DIR}/compile-testing/header-test.cc @ONLY)

    try_compile(RET
        ${CMAKE_BINARY_DIR}/compile-testing
        ${CMAKE_BINARY_DIR}/compile-testing/header-test.cc
        CMAKE_FLAGS ${CMAKE_CXX_FLAGS}
        INCLUDE_DIRECTORIES ${dirs}
        OUTPUT_VARIABLE RET_OUT)

    IF(RET)
        set(${VAR} ON PARENT_SCOPE)
    ELSE()
        set(${VAR} OFF PARENT_SCOPE)
    ENDIF()

endfunction(CHECK_FOR_HEADER VAR HEADER)

#==============================================================================#

SET_MACRO_FIELDS(DEFINE
    LABEL   "F77_FUNC"
    MACRO   "F77_FUNC(name)"
    ENTRY   "name ## _"
    MSG     "to a macro mangling the given Fortran function name")

#==============================================================================#

GET_DEFINED_FLAG(HAS USE_BLAS)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_BLAS"
    ENTRY   "1"
    MSG     "if you have a BLAS library")

#==============================================================================#

GET_DEFINED_FLAG(HAS USE_LAPACK)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_LAPACK"
    ENTRY   "1"
    MSG     "if you have a LAPACK library")

#==============================================================================#

GET_DEFINED_FLAG(HAS USE_OPENBLAS)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_OPENBLAS"
    ENTRY   "1"
    MSG     "if you have a OpenBLAS library")

#==============================================================================#

GET_DEFINED_FLAG(HAS IMF_FOUND)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_MATH"
    ENTRY   "1"
    MSG     "if you have the Intel IMF math library")

#==============================================================================#

CHECK_FOR_HEADER(HEADER_FILE "math.h")
GET_DEFINED_FLAG(HAS HEADER_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_MATH_H"
    ENTRY   "1"
    MSG     "if you have the <math.h> library")

#==============================================================================#

CHECK_FOR_HEADER(HEADER_FILE "cstdlib")
GET_DEFINED_FLAG(HAS HEADER_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "STDC_HEADERS"
    ENTRY   "1"
    MSG     "to 1 if you have the ANSI C header files.")

#==============================================================================#

CHECK_FOR_HEADER(HEADER_FILE "memory.h")
GET_DEFINED_FLAG(HAS HEADER_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_MEMORY_H"
    ENTRY   "1"
    MSG     "if you have the <memory.h> library")

#==============================================================================#

CHECK_FOR_HEADER(HEADER_FILE "inttypes.h")
GET_DEFINED_FLAG(HAS HEADER_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_INTTYPES_H"
    ENTRY   "1"
    MSG     "to 1 if you have the <inttypes.h> header file.")

#==============================================================================#

CHECK_FOR_HEADER(HEADER_FILE "stdint.h")
GET_DEFINED_FLAG(HAS HEADER_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_STDINT_H"
    ENTRY   "1"
    MSG     "to 1 if you have the <stdint.h> header file.")

#==============================================================================#

CHECK_FOR_HEADER(HEADER_FILE "stdlib.h")
GET_DEFINED_FLAG(HAS HEADER_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_STDLIB_H"
    ENTRY   "1"
    MSG     "to 1 if you have the <stdlib.h> header file.")

#==============================================================================#

CHECK_FOR_HEADER(HEADER_FILE "strings.h")
GET_DEFINED_FLAG(HAS HEADER_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_STRINGS_H"
    ENTRY   "1"
    MSG     "to 1 if you have the <strings.h> header file.")

#==============================================================================#

CHECK_FOR_HEADER(HEADER_FILE "string.h")
GET_DEFINED_FLAG(HAS HEADER_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_STRING_H"
    ENTRY   "1"
    MSG     "to 1 if you have the <string.h> header file.")

#==============================================================================#

CHECK_FOR_HEADER(HEADER_FILE "sys/stat.h")
GET_DEFINED_FLAG(HAS HEADER_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_SYS_STAT_H"
    ENTRY   "1"
    MSG     "to 1 if you have the <sys/stat.h> header file.")

#==============================================================================#

CHECK_FOR_HEADER(HEADER_FILE "sys/types.h")
GET_DEFINED_FLAG(HAS HEADER_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_SYS_TYPES_H"
    ENTRY   "1"
    MSG     "to 1 if you have the <sys/types.h> header file.")

#==============================================================================#

CHECK_FOR_HEADER(HEADER_FILE "unistd.h")
GET_DEFINED_FLAG(HAS HEADER_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_UNISTD_H"
    ENTRY   "1"
    MSG     "to 1 if you have the <unistd.h> header file.")

#==============================================================================#

GET_DEFINED_FLAG(HAS wcslib_FOUND)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_WCSLIB"
    ENTRY   "1"
    MSG     "if you are using wcslib")

#==============================================================================#

GET_DEFINED_FLAG(HAS aatm_FOUND)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_AATM"
    ENTRY   "1"
    MSG     "if you are using aatm")

#==============================================================================#

CHECK_FOR_HEADER(HEADER_FILE "wcslib/wcs.h")
GET_DEFINED_FLAG(HAS HEADER_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_WCSLIB_WCS_H"
    ENTRY   "1"
    MSG     "to 1 if you have the <wcslib/wcs.h> header file.")

#==============================================================================#

GET_DEFINED_FLAG(HAS USE_MKL)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_MKL"
    ENTRY   "1"
    MSG     "if you are using MKL")

#==============================================================================#

GET_DEFINED_FLAG(HAS USE_TBB)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_TBB"
    ENTRY   "1"
    MSG     "if you are using TBB")

#==============================================================================#

GET_DEFINED_FLAG(HAS Threads_FOUND)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_PTHREAD"
    ENTRY   "1"
    MSG     "if you have POSIX threads libraries and header files. ")

#==============================================================================#

GET_DEFINED_FLAG(HAS OpenMP_FOUND)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_OPENMP"
    ENTRY   "1"
    MSG     "if OpenMP is enabled.")

#==============================================================================#

CHECK_FOR_HEADER(MKL_DFTI_FILE "mkl_dfti.h")
GET_DEFINED_FLAG(HAS MKL_DFTI_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_MKL_DFTI_H"
    ENTRY   "1"
    MSG     "if to 1 if you have the <mkl_dfti.h> header file.")

#==============================================================================#

GET_DEFINED_FLAG(HAS Python3_FOUND)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_PYTHON"
    ENTRY   "\"${PYTHON_VERSION_STRING}\""
    MSG     "contains the Python version number currently in use")

#==============================================================================#

CHECK_FOR_HEADER(HEADER_FILE "dlfcn.h")
GET_DEFINED_FLAG(HAS HEADER_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_DLFCN_H"
    ENTRY   1
    MSG     "to 1 if you have the <dlfcn.h> header file.")

#==============================================================================#

GET_DEFINED_FLAG(HAS USE_ELEMENTAL)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_ELEMENTAL"
    ENTRY   1
    MSG     "we have Elemental")

#==============================================================================#

CHECK_FOR_HEADER(HEADER_FILE "El.hpp")
GET_DEFINED_FLAG(HAS HEADER_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_EL_HPP"
    ENTRY   1
    MSG     "to 1 if you have the <El.hpp> header file.")

#==============================================================================#

GET_DEFINED_FLAG(HAS USE_FFTW)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_FFTW"
    ENTRY   1
    MSG     "we have FFTW")

#==============================================================================#

GET_DEFINED_FLAG(HAS FFTW3_THREADS_FOUND)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_FFTW_THREADS"
    ENTRY   1
    MSG     "if you have the FFTW threads library.")

#==============================================================================#

CHECK_FOR_HEADER(HEADER_FILE "fftw3.h")
GET_DEFINED_FLAG(HAS HEADER_FILE)
SET_MACRO_FIELDS(${HAS}
    LABEL   "HAVE_FFTW3_H"
    ENTRY   1
    MSG     "to 1 if you have the <fftw3.h> header file.")

#==============================================================================#
#==============================================================================#
#
#           PACKAGE DETAILS
#
#==============================================================================#
#==============================================================================#

execute_process(COMMAND ${GIT_COMMAND} describe --tags --abbrev=0
    OUTPUT_VARIABLE GIT_LAST_TAG
    OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

execute_process(COMMAND ${GIT_COMMAND} rev-list --count --branches
    OUTPUT_VARIABLE GIT_DEV_NUMBER
    OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

execute_process(COMMAND ${GIT_COMMAND} remote get-url origin
    OUTPUT_VARIABLE GIT_URL
    OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

STRING(REPLACE "/" ";" GIT_TARNAME "${GIT_URL}")
LIST(LENGTH GIT_TARNAME GIT_TARNAME_LENGTH)
MATH(EXPR GIT_TARNAME_LENGTH "${GIT_TARNAME_LENGTH}-1")
LIST(GET GIT_TARNAME ${GIT_TARNAME_LENGTH} GIT_TARNAME)
STRING(REPLACE ".git" "" GIT_TARNAME "${GIT_TARNAME}")

#==============================================================================#

STRING(TOLOWER "${CMAKE_PROJECT_NAME}" LOWER_CMAKE_PROJECT_NAME)
SET_MACRO_FIELDS(DEFINE
    LABEL   "PACKAGE"
    ENTRY   "\"${LOWER_CMAKE_PROJECT_NAME}\""
    MSG     "of package")

#==============================================================================#

SET_MACRO_FIELDS(DEFINE
    LABEL   "PACKAGE_BUGREPORT"
    ENTRY   "\"${PROJECT_BUGREPORT}\""
    MSG     "to the address where bug reports for this package should be sent.")

#==============================================================================#

SET_MACRO_FIELDS(DEFINE
    LABEL   "PACKAGE_NAME"
    ENTRY   "\"${CMAKE_PROJECT_NAME}\""
    MSG     "to the full name of the package")

#==============================================================================#

SET_MACRO_FIELDS(DEFINE
    LABEL   "PACKAGE_STRING"
    ENTRY   "\"${CMAKE_PROJECT_NAME} ${GIT_LAST_TAG}.dev${GIT_DEV_NUMBER}\""
    MSG     "to the version of this package.")

#==============================================================================#

SET_MACRO_FIELDS(DEFINE
    LABEL   "PACKAGE_VERSION"
    ENTRY   "\"${GIT_LAST_TAG}.dev${GIT_DEV_NUMBER}\""
    MSG     "to the version of this package.")

#==============================================================================#

SET_MACRO_FIELDS(DEFINE
    LABEL   "PACKAGE_URL"
    ENTRY   "\"${GIT_URL}\""
    MSG     "to the home page for this package.")

#==============================================================================#

SET_MACRO_FIELDS(DEFINE
    LABEL   "PACKAGE_TARNAME"
    ENTRY   "\"${GIT_TARNAME}\""
    MSG     "to the one symbol short name of this package.")

#==============================================================================#

SET_MACRO_FIELDS(DEFINE
    LABEL   "VERSION"
    ENTRY   "\"${GIT_LAST_TAG}.dev${GIT_DEV_NUMBER}\""
    MSG     "to the version number of this package.")

#==============================================================================#




#==============================================================================#
#==============================================================================#
#
#       GENERATE CONFIG FILE
#
#==============================================================================#
#==============================================================================#
STRING(TOLOWER "${PROJECT_NAME}" LOWER_PROJECT_NAME)

# write a build specific config.h header file
write_toast_config_file(
    TEMPLATE
        ${CMAKE_SOURCE_DIR}/cmake/Templates/config.hh.in
    FILE_NAME
        config.h
    WRITE_PATH
        ${PROJECT_BINARY_DIR}/config
    INSTALL_PATH
        ${CMAKE_INSTALL_CMAKEDIR}/${CMAKE_BUILD_TYPE}/${LOWER_PROJECT_NAME}
    DEFINE_VARS
        "${DEFINE_VARIABLES}"
    UNDEFINE_VARS
        "${UNDEFINE_VARIABLES}"
    SKIP_VARS
        ${SKIP_DEFINITIONS_IN_CONFIG}
)


cmake_policy(POP)


