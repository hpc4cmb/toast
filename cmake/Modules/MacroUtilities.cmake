# MacroUtilities - useful macros and functions for generic tasks
#
# CMake Extensions
# ----------------
# macro set_ifnot(<var> <value>)
#       If variable var is not set, set its value to that provided
#
# function enum_option(<option>
#                      VALUES <value1> ... <valueN>
#                      TYPE   <valuetype>
#                      DOC    <docstring>
#                      [DEFAULT <elem>]
#                      [CASE_INSENSITIVE])
#          Declare a cache variable <option> that can only take values
#          listed in VALUES. TYPE may be FILEPATH, PATH or STRING.
#          <docstring> should describe that option, and will appear in
#          the interactive CMake interfaces. If DEFAULT is provided,
#          <elem> will be taken as the zero-indexed element in VALUES
#          to which the value of <option> should default to if not
#          provided. Otherwise, the default is taken as the first
#          entry in VALUES. If CASE_INSENSITIVE is present, then
#          checks of the value of <option> against the allowed values
#          will ignore the case when performing string comparison.
#
#
# General
# --------------
# function add_feature(<NAME> <DOCSTRING>)
#          Add a  feature, whose activation is specified by the
#          existence of the variable <NAME>, to the list of enabled/disabled
#          features, plus a docstring describing the feature
#
# function print_enabled_features()
#          Print enabled  features plus their docstrings.
#
#

# - Include guard
if(__macroutilities_isloaded)
  return()
endif()
set(__macroutilities_isloaded YES)

cmake_policy(PUSH)
if(NOT CMAKE_VERSION VERSION_LESS 3.1)
    cmake_policy(SET CMP0054 NEW)
endif()

#-----------------------------------------------------------------------
# CMAKE EXTENSIONS
#-----------------------------------------------------------------------
# macro set_ifnot(<var> <value>)
#       If variable var is not set, set its value to that provided
#
macro(set_ifnot _var _value)
  if(NOT DEFINED ${_var})
    set(${_var} ${_value})
  endif()
endmacro()

#-----------------------------------------------------------------------
# macro set_ifnot_match(<var> <value>)
#       If variable var is not set, set its value to that provided
#
macro(SET_IFNOT_MATCH VAR APPEND)
    if(NOT "${APPEND}" STREQUAL "")
        STRING(REGEX MATCH "${APPEND}" _MATCH "${${VAR}}")
        if(NOT "${_MATCH}" STREQUAL "")
            SET(${VAR} "${${VAR}} ${APPEND}")
        endif()
    endif()
endmacro()

#-----------------------------------------------------------------------
# macro cache_ifnot(<var> <value>)
#       If variable var is not set, set its value to that provided and cache it
#
macro(cache_ifnot _var _value _type _doc)
  if(NOT ${_var} OR NOT ${CACHE_VARIABLES} MATCHES ${_var})
    set(${_var} ${_value} CACHE ${_type} "${_doc}")
  endif()
endmacro()

#-----------------------------------------------------------------------
# function enum_option(<option>
#                      VALUES <value1> ... <valueN>
#                      TYPE   <valuetype>
#                      DOC    <docstring>
#                      [DEFAULT <elem>]
#                      [CASE_INSENSITIVE])
#          Declare a cache variable <option> that can only take values
#          listed in VALUES. TYPE may be FILEPATH, PATH or STRING.
#          <docstring> should describe that option, and will appear in
#          the interactive CMake interfaces. If DEFAULT is provided,
#          <elem> will be taken as the zero-indexed element in VALUES
#          to which the value of <option> should default to if not
#          provided. Otherwise, the default is taken as the first
#          entry in VALUES. If CASE_INSENSITIVE is present, then
#          checks of the value of <option> against the allowed values
#          will ignore the case when performing string comparison.
#
function(enum_option _var)
  set(options CASE_INSENSITIVE)
  set(oneValueArgs DOC TYPE DEFAULT)
  set(multiValueArgs VALUES)
  cmake_parse_arguments(_ENUMOP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # - Validation as needed arguments
  if(NOT _ENUMOP_VALUES)
    message(FATAL_ERROR "enum_option must be called with non-empty VALUES\n(Called for enum_option '${_var}')")
  endif()

  # - Set argument defaults as needed
  if(_ENUMOP_CASE_INSENSITIVE)
    set(_ci_values )
    foreach(_elem ${_ENUMOP_VALUES})
      string(TOLOWER "${_elem}" _ci_elem)
      list(APPEND _ci_values "${_ci_elem}")
    endforeach()
    set(_ENUMOP_VALUES ${_ci_values})
  endif()

  set_ifnot(_ENUMOP_TYPE STRING)
  set_ifnot(_ENUMOP_DEFAULT 0)
  list(GET _ENUMOP_VALUES ${_ENUMOP_DEFAULT} _default)

  if(NOT DEFINED ${_var})
    set(${_var} ${_default} CACHE ${_ENUMOP_TYPE} "${_ENUMOP_DOC} (${_ENUMOP_VALUES})")
  else()
    set(_var_tmp ${${_var}})
    if(_ENUMOP_CASE_INSENSITIVE)
      string(TOLOWER ${_var_tmp} _var_tmp)
    endif()

    list(FIND _ENUMOP_VALUES ${_var_tmp} _elem)
    if(_elem LESS 0)
      message(FATAL_ERROR "Value '${${_var}}' for variable ${_var} is not allowed\nIt must be selected from the set: ${_ENUMOP_VALUES} (DEFAULT: ${_default})\n")
    else()
      # - convert to lowercase
      if(_ENUMOP_CASE_INSENSITIVE)
        set(${_var} ${_var_tmp} CACHE ${_ENUMOP_TYPE} "${_ENUMOP_DOC} (${_ENUMOP_VALUES})" FORCE)
      endif()
    endif()
  endif()
endfunction()

#-----------------------------------------------------------------------
# from
#   http://www.cmake.org/pipermail/cmake/2008-April/021345.html
#-----------------------------------------------------------------------
#
#
# Adds a file or directory to the FILES_TO_DELETE var so that it is removed
# when "make distclean" is run
#
# Prototype:
#    ADD_TO_DISTCLEAN(file)
# Parameters:
#    file    A file or dir
#

MACRO(ADD_TO_DISTCLEAN TARGET_TO_DELETE)
     SET( FILES_TO_DELETE ${FILES_TO_DELETE} ${TARGET_TO_DELETE} )
ENDMACRO(ADD_TO_DISTCLEAN)

#-----------------------------------------------------------------------
# from
#   http://www.cmake.org/pipermail/cmake/2008-April/021345.html
#-----------------------------------------------------------------------
#
# Create a "make distclean" target
#
# Prototype:
#    GENERATE_DISTCLEAN_TARGET()
# Parameters:
#    (none)
#

MACRO(GENERATE_DISTCLEAN_TARGET)
    IF( EXISTS ${PROJECT_BINARY_DIR}/distclean_manifest.txt )
        FILE( REMOVE ${PROJECT_BINARY_DIR}/distclean_manifest.txt )
    ENDIF( EXISTS ${PROJECT_BINARY_DIR}/distclean_manifest.txt )

    IF(MSVC)
        SET( FILES_TO_DELETE ${FILES_TO_DELETE} ${PROJECT_BINARY_DIR}/*.dir )
        ADD_TO_DISTCLEAN( ${PROJECT_BINARY_DIR}/*.vcproj )
        ADD_TO_DISTCLEAN( ${PROJECT_BINARY_DIR}/*.vcproj.cmake )
        ADD_TO_DISTCLEAN( ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.sln )
    ENDIF(MSVC)
    ADD_TO_DISTCLEAN( ${PROJECT_BINARY_DIR}/distclean.dir )
    ADD_TO_DISTCLEAN( ${PROJECT_BINARY_DIR}/CMakeCache.txt )
    ADD_TO_DISTCLEAN( ${PROJECT_BINARY_DIR}/install_manifest.txt )
    ADD_TO_DISTCLEAN( ${PROJECT_BINARY_DIR}/CPackConfig.cmake )
    ADD_TO_DISTCLEAN( ${PROJECT_BINARY_DIR}/CPackSourceConfig.cmake )
    ADD_TO_DISTCLEAN( ${PROJECT_BINARY_DIR}/_CPack_Packages )
    ADD_TO_DISTCLEAN( ${PROJECT_BINARY_DIR}/PACKAGE.dir )
    ADD_TO_DISTCLEAN( ${PROJECT_BINARY_DIR}/cmake_install.cmake )

    # This code does not work yet. I don't know why CPACK_GENERATOR is null.
    IF(CPACK_GENERATOR)
        FOREACH(gen ${CPACK_GENERATOR})
            MESSAGE("Adding file ${PROJECT_BINARY_DIR}/${CPACK_SOURCE_PACKAGE_FILE_NAME}.${gen} to distclean")
            ADD_TO_DISTCLEAN(${PROJECT_BINARY_DIR}/${CPACK_SOURCE_PACKAGE_FILE_NAME}.${gen} )
        ENDFOREACH(gen)
    ELSE(CPACK_GENERATOR)
        MESSAGE("CPACK_GENERATOR was not defined (value: ${CPACK_GENERATOR})")
    ENDIF(CPACK_GENERATOR)



    IF(EXECUTABLE_OUTPUT_PATH)
        ADD_TO_DISTCLEAN( ${EXECUTABLE_OUTPUT_PATH} )
    ENDIF(EXECUTABLE_OUTPUT_PATH)
    IF(LIBRARY_OUTPUT_PATH)
        ADD_TO_DISTCLEAN( ${LIBRARY_OUTPUT_PATH} )
    ENDIF(LIBRARY_OUTPUT_PATH)
    ADD_TO_DISTCLEAN( ${PROJECT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY} )
    ADD_TO_DISTCLEAN( ${PROJECT_BINARY_DIR}/Makefile )
    ADD_TO_DISTCLEAN( ${PROJECT_BINARY_DIR}/distclean_manifest.txt )

    FOREACH(f ${FILES_TO_DELETE})
        FILE(TO_NATIVE_PATH ${f} ff)
        FILE(APPEND ${PROJECT_BINARY_DIR}/distclean_manifest.txt ${ff})
        FILE(APPEND ${PROJECT_BINARY_DIR}/distclean_manifest.txt "\n")
    ENDFOREACH(f)

    IF(WIN32)
        FILE( TO_NATIVE_PATH ${PROJECT_BINARY_DIR} PROJECT_BINARY_DIR_WIN32 )
        ADD_CUSTOM_TARGET(distclean
                          FOR /F \"tokens=1* delims= \" %%f IN
                            \(${PROJECT_BINARY_DIR_WIN32}\\distclean_manifest.txt\) DO \(
                            IF EXIST %%f\\nul \(
                                rd /q /s %%f
                            \) ELSE \(
                                IF EXIST %%f \(
                                    del /q /f %%f
                                \) ELSE \(
                                    echo Warning: Problem when removing %%f. - Probable causes: File already removed or not enough permissions
                                \)
                            \)
                        \)
        )
    ELSE(WIN32)
        # Unix
        ADD_CUSTOM_TARGET(distclean cat
                          "${PROJECT_BINARY_DIR}/distclean_manifest.txt" | while read f \; do if
                            \[ -e \"\$\${f}\" \]; then rm -rf \"\$\${f}\" \; else echo \"Warning:
                            Problem when removing \"\$\${f}\" - Probable causes: File already
                            removed or not enough permissions\" \; fi\; done COMMENT Cleaning all
                            generated files...
        )
    ENDIF(WIN32)
ENDMACRO(GENERATE_DISTCLEAN_TARGET)


#-----------------------------------------------------------------------
# Determine if two paths are the same
#
#-----------------------------------------------------------------------
function(equal_paths VAR PATH1 PATH2)
    get_filename_component(PATH1 ${PATH1} ABSOLUTE)
    get_filename_component(PATH2 ${PATH2} ABSOLUTE)

    if ("${PATH1}" STREQUAL "${PATH2}")
        set(${VAR} ON PARENT_SCOPE)
    else()
        set(${VAR} OFF PARENT_SCOPE)
    endif()
endfunction(equal_paths VAR PATH1 PATH2)


#-----------------------------------------------------------------------
# Resolve symbolic links, remove duplicates, and remove system paths
#   in CMAKE_PREFIX_PATH
#
#-----------------------------------------------------------------------
function(clean_prefix_path)
    set(_prefix_path )
    foreach(_path ${CMAKE_PREFIX_PATH})
        get_filename_component(_path ${_path} REALPATH)
        set(IS_SYS_PATH OFF)
        # loop over system path types
        foreach(_type PREFIX INCLUDE LIBRARY APPBUNDLE FRAMEWORK PROGRAM)
            # loop over system paths
            foreach(_syspath ${CMAKE_SYSTEM_${_type}_PATH})
                # check if equal
                equal_paths(IS_SYS_PATH ${_path} ${_syspath})
                # exit loop if equal
                if(IS_SYS_PATH)
                    break()
                endif(IS_SYS_PATH)
            endforeach(_syspath ${CMAKE_SYSTEM_${_type}_PATH})
            # exit loop if equal
            if(IS_SYS_PATH)
                break()
            endif(IS_SYS_PATH)
        endforeach(_type PREFIX INCLUDE LIBRARY APPBUNDLE FRAMEWORK PROGRAM)
        # if not a system path
        if(NOT IS_SYS_PATH)
            list(APPEND _prefix_path ${_path})
        endif(NOT IS_SYS_PATH)
    endforeach()
    # remove any duplicates
    if(NOT "${_prefix_path}" STREQUAL "")
        list(REMOVE_DUPLICATES _prefix_path)
    endif()
    # force the new prefix path
    set(CMAKE_PREFIX_PATH ${_prefix_path} CACHE PATH
        "Prefix path for finding packages" FORCE)
endfunction()


#-----------------------------------------------------------------------
# Add a defined ${PACKAGE_NAME}_ROOT variable defined via CMake or in
# environment to the CMAKE_PREFIX_PATH
#
# This macro should NOT be called directly, instead call
# ConfigureRootSearchPath (i.e. with "_" prefix)
#-----------------------------------------------------------------------
function(subConfigureRootSearchPath _package_name _search_other)

    mark_as_advanced(PREVIOUS_${_package_name}_ROOT)

    # if ROOT not already defined and ENV variable defines it
    if(NOT ${_package_name}_ROOT AND
       NOT "$ENV{${_package_name}_ROOT}" STREQUAL "")
        cache_ifnot(${_package_name}_ROOT $ENV{${_package_name}_ROOT}
                    FILEPATH "ROOT search path for ${_package_name}")
    endif()

    # if ROOT is still not defined, check upper case and capitalize version
    # and return
    if(NOT ${_package_name}_ROOT)
        if(_search_other)
            string(TOUPPER "${_package_name}" ALT_PACKAGE_NAME)
            if("${ALT_PACKAGE_NAME}" STREQUAL "${_package_name}")
                capitalize("${_package_name}" ALT_PACKAGE_NAME)
                subConfigureRootSearchPath(${ALT_PACKAGE_NAME} OFF)
            else()
                subConfigureRootSearchPath(${ALT_PACKAGE_NAME} OFF)
            endif()
        endif()
        return()
    endif()

    # if ROOT is defined and has changed
    if(${_package_name}_ROOT AND PREVIOUS_${_package_name}_ROOT AND
       NOT "${PREVIOUS_${_package_name}_ROOT}" STREQUAL "${${_package_name}_ROOT}")
        if(NOT "${PREVIOUS_${_package_name}_ROOT}" STREQUAL "${${_package_name}_ROOT}")
            set(UNCACHE_PACKAGE_VARS ON)
        endif()
        # make sure it exists and is a directory
        if(EXISTS "${${_package_name}_ROOT}" AND
           IS_DIRECTORY "${${_package_name}_ROOT}")
            set(CMAKE_PREFIX_PATH ${${_package_name}_ROOT} ${CMAKE_PREFIX_PATH}
                CACHE PATH "CMake prefix paths" FORCE)
            # store previous root to see if it changed
            set(PREVIOUS_${_package_name}_ROOT ${${_package_name}_ROOT}
                CACHE FILEPATH "Previous root search path for ${_package_name}" FORCE)
            clean_prefix_path()
        else()
            message(WARNING "${_package_name}_ROOT specified an invalid directory")
            unset(${_package_name}_ROOT CACHE)
        endif()
        # root changed so we want to refind the package
        if(UNCACHE_PACKAGE_VARS)
            string(TOLOWER "${_package_name}" CAP_PACKAGE_NAME)
            string(SUBSTRING "${CAP_PACKAGE_NAME}" 0 1 FIRST)
            string(SUBSTRING "${CAP_PACKAGE_NAME}" 1 -1 REST)
            string(TOUPPER "${FIRST}" FIRST)
            string(CONCAT CAP_PACKAGE_NAME "${FIRST}" "${REST}")
            string(TOUPPER "${_package_name}" UPP_PACKAGE_NAME)
            string(TOLOWER "${_package_name}" LOW_PACKAGE_NAME)

            get_cmake_property(CACHED_VARIABLES CACHE_VARIABLES)
            foreach(_name ${_package_name} ${CAP_PACKAGE_NAME}
                    ${UPP_PACKAGE_NAME} ${LOW_PACKAGE_NAME})
                # skip maintain
                if(NOT MAINTAIN_${_name}_CACHE)
                    # loop over cached variables
                    foreach(_var ${CACHED_VARIABLES})
                        if("${_var}_" MATCHES "${_name}")
                            if(NOT "${_var}" STREQUAL "${_name}_ROOT" AND
                                    NOT "${_var}" STREQUAL "PREVIOUS_${_name}_ROOT" AND
                                    NOT "${_var}" STREQUAL "MAINTAIN_${_name}_CACHE")
                                unset(${_var} CACHE)
                                list(REMOVE_ITEM CACHED_VARIABLES "${_var}")
                            endif()
                        endif()
                    endforeach(_var ${CACHE_VARIABLES})
                endif()
            endforeach()
        endif()
    else()
        if(EXISTS "${${_package_name}_ROOT}" AND
                IS_DIRECTORY "${${_package_name}_ROOT}")
            set(_add ON)
            foreach(_path ${CMAKE_PREFIX_PATH})
                if("${_path}" STREQUAL "${${_package_name}_ROOT}")
                    set(_add OFF)
                    break()
                endif()
            endforeach()
            if(_add)
                set(CMAKE_PREFIX_PATH ${${_package_name}_ROOT} ${CMAKE_PREFIX_PATH}
                    CACHE PATH "CMake prefix paths" FORCE)
                clean_prefix_path()
            endif()
            # store previous root to see if it changed
            set(PREVIOUS_${_package_name}_ROOT ${${_package_name}_ROOT}
                CACHE FILEPATH "Previous root search path for ${_package_name}" FORCE)
        else()
            message(WARNING "${_package_name}_ROOT specified an invalid directory")
            unset(${_package_name}_ROOT CACHE)
        endif()
    endif()

endfunction()


#-----------------------------------------------------------------------
# Add a defined ${PACKAGE_NAME}_ROOT variable defined via CMake or in
# environment to the CMAKE_PREFIX_PATH
#
# Different from _ConfigureRootSearchPath
#-----------------------------------------------------------------------
macro(ConfigureRootSearchPath)
    foreach(_package_name ${ARGN})
        subConfigureRootSearchPath(${_package_name} ON)
        if($ENV{${_package_name}_DIR})
            set(${_package_name}_DIR "$ENV{${_package_name}_DIR}" CACHE PATH
                "CMake config directory for ${_package_name}")
        endif()
    endforeach()
endmacro()

#-----------------------------------------------------------------------
# GENERAL
#-----------------------------------------------------------------------
# function add_feature(<NAME> <DOCSTRING>)
#          Add a project feature, whose activation is specified by the
#          existence of the variable <NAME>, to the list of enabled/disabled
#          features, plus a docstring describing the feature
#
function(ADD_FEATURE _var _description)
  set(EXTRA_DESC "")
  foreach(currentArg ${ARGN})
      if(NOT "${currentArg}" STREQUAL "${_var}" AND
         NOT "${currentArg}" STREQUAL "${_description}")
          set(EXTRA_DESC "${EXTA_DESC}${currentArg}")
      endif()
  endforeach()

  set_property(GLOBAL APPEND PROPERTY PROJECT_FEATURES ${_var})
  #set(${_var} ${${_var}} CACHE INTERNAL "${_description}${EXTRA_DESC}")

  set_property(GLOBAL PROPERTY ${_var}_DESCRIPTION "${_description}${EXTRA_DESC}")
endfunction()

#------------------------------------------------------------------------------#
# function add_subfeature(<ROOT_OPTION> <NAME> <DOCSTRING>)
#          Add a subfeature, whose activation is specified by the
#          existence of the variable <NAME>, to the list of enabled/disabled
#          features, plus a docstring describing the feature
#
function(ADD_SUBFEATURE _root _var _description)
    set(EXTRA_DESC "")
    foreach(currentArg ${ARGN})
        if(NOT "${currentArg}" STREQUAL "${_var}" AND
           NOT "${currentArg}" STREQUAL "${_description}")
            set(EXTRA_DESC "${EXTA_DESC}${currentArg}")
        endif()
    endforeach()

    set_property(GLOBAL APPEND PROPERTY ${_root}_FEATURES ${_var})
    set_property(GLOBAL PROPERTY ${_root}_${_var}_DESCRIPTION "${_description}${EXTRA_DESC}")
    #set(${_var} ${${_var}} CACHE INTERNAL "${_description}${EXTRA_DESC}")

endfunction()


#------------------------------------------------------------------------------#
# function add_option(<OPTION_NAME> <DOCSRING> <DEFAULT_SETTING> [NO_FEATURE])
#          Add an option and add as a feature if NO_FEATURE is not provided
#
MACRO(ADD_OPTION _NAME _MESSAGE _DEFAULT)
    SET(_FEATURE ${ARGN})
    OPTION(${_NAME} "${_MESSAGE}" ${_DEFAULT})
    IF(NOT "${_FEATURE}" STREQUAL "NO_FEATURE")
        ADD_FEATURE(${_NAME} "${_MESSAGE}")
    ELSE()
        MARK_AS_ADVANCED(${_NAME})
    ENDIF()
ENDMACRO()


#------------------------------------------------------------------------------#
# function print_enabled_features()
#          Print enabled  features plus their docstrings.
#
function(print_enabled_features)
    set(_basemsg "The following features are defined/enabled (+):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_features GLOBAL PROPERTY PROJECT_FEATURES)
    if(NOT "${_features}" STREQUAL "")
        list(REMOVE_DUPLICATES _features)
        list(SORT _features)
    endif()
    foreach(_feature ${_features})
        if(${_feature})
            # add feature to text
            set(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")
            # get description
            get_property(_desc GLOBAL PROPERTY ${_feature}_DESCRIPTION)
            # print description, if not standard ON/OFF, print what is set to
            if(_desc)
                if(NOT "${${_feature}}" STREQUAL "ON" AND
                   NOT "${${_feature}}" STREQUAL "TRUE")
                    set(_currentFeatureText "${_currentFeatureText}: ${_desc} -- [\"${${_feature}}\"]")
                else()
                    set(_currentFeatureText "${_currentFeatureText}: ${_desc}")
                endif()
                set(_desc NOTFOUND)
            endif(_desc)
            # check for subfeatures
            get_property(_subfeatures GLOBAL PROPERTY ${_feature}_FEATURES)
            # remove duplicates and sort if subfeatures exist
            if(NOT "${_subfeatures}" STREQUAL "")
                list(REMOVE_DUPLICATES _subfeatures)
                list(SORT _subfeatures)
            endif()

            # sort enabled and disabled features into lists
            set(_enabled_subfeatures )
            set(_disabled_subfeatures )
            foreach(_subfeature ${_subfeatures})
                if(${_subfeature})
                    list(APPEND _enabled_subfeatures ${_subfeature})
                else()
                    list(APPEND _disabled_subfeatures ${_subfeature})
                endif()
            endforeach()

            # print that following are enabled subfeatures
            #if(_enabled_subfeatures)
            #    set(_currentFeatureText "${_currentFeatureText}\n\t-- The following subfeatures of ${_feature} are defined/enabled:")
            #endif()

            # loop over enabled subfeatures
            foreach(_subfeature ${_enabled_subfeatures})
                # add subfeature to text
                set(_currentFeatureText "${_currentFeatureText}\n       + ${_subfeature}")
                # get subfeature description
                get_property(_subdesc GLOBAL PROPERTY ${_feature}_${_subfeature}_DESCRIPTION)
                # print subfeature description. If not standard ON/OFF, print
                # what is set to
                if(_subdesc)
                    if(NOT "${${_subfeature}}" STREQUAL "ON" AND
                       NOT "${${_subfeature}}" STREQUAL "TRUE")
                        set(_currentFeatureText "${_currentFeatureText}: ${_subdesc} -- [\"${${_subfeature}}\"]")
                    else()
                        set(_currentFeatureText "${_currentFeatureText}: ${_subdesc}")
                    endif()
                    set(_subdesc NOTFOUND)
                endif(_subdesc)
            endforeach(_subfeature)

            # print that following are disabled subfeatures
            #if(_disabled_subfeatures)
            #    set(_currentFeatureText "${_currentFeatureText}\n\t-- The following subfeatures of ${_feature} are NOT defined/enabled:")
            #endif()

            # loop over disabled subfeatures
            foreach(_subfeature ${_disabled_subfeatures})
                # add subfeature to text
                set(_currentFeatureText "${_currentFeatureText}\n       - ${_subfeature}")
                # get subfeature description
                get_property(_subdesc GLOBAL PROPERTY ${_feature}_${_subfeature}_DESCRIPTION)
                # print subfeature description.
                if(_subdesc)
                    set(_currentFeatureText "${_currentFeatureText}: ${_subdesc}")
                    set(_subdesc NOTFOUND)
                endif(_subdesc)
            endforeach(_subfeature)

        endif(${_feature})
    endforeach(_feature)

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
endfunction()


#------------------------------------------------------------------------------#
# function print_disabled_features()
#          Print disabled features plus their docstrings.
#
function(print_disabled_features)
    set(_basemsg "The following features are NOT defined/enabled (-):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_features GLOBAL PROPERTY PROJECT_FEATURES)
    if(NOT "${_features}" STREQUAL "")
        list(REMOVE_DUPLICATES _features)
        list(SORT _features)
    endif()
    foreach(_feature ${_features})
        if(NOT ${_feature})
            set(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")

            get_property(_desc GLOBAL PROPERTY ${_feature}_DESCRIPTION)

            if(_desc)
              set(_currentFeatureText "${_currentFeatureText}: ${_desc}")
              set(_desc NOTFOUND)
            endif(_desc)
        endif()
    endforeach(_feature)

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
endfunction()

#------------------------------------------------------------------------------#
# function print_features()
#          Print all features plus their docstrings.
#
function(print_features)
    message("")
    print_enabled_features()
    print_disabled_features()
endfunction()

#------------------------------------------------------------------------------#
macro(DETERMINE_LIBDIR_DEFAULT)
  set(LIBDIR_DEFAULT "lib")
  # Override this default 'lib' with 'lib64' if:
  #  - we are on Linux system but NOT cross-compiling
  #  - we are NOT on debian
  #  - we are on a 64 bits system
  # reason is: amd64 ABI: http://www.x86-64.org/documentation/abi.pdf
  # Note that the future of multi-arch handling may be even
  # more complicated than that: http://wiki.debian.org/Multiarch
  if(DEFINED _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX)
      set(__LAST_LIBDIR_DEFAULT "lib")
      # __LAST_LIBDIR_DEFAULT is the default value that we compute from
      # _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX, not a cache entry for
      # the value that was last used as the default.
      # This value is used to figure out whether the user changed the
      # CMAKE_INSTALL_LIBDIR value manually, or if the value was the
      # default one. When CMAKE_INSTALL_PREFIX changes, the value is
      # updated to the new default, unless the user explicitly changed it.
  endif()
  if(CMAKE_SYSTEM_NAME MATCHES "^(Linux|kFreeBSD|GNU)$"
      AND NOT CMAKE_CROSSCOMPILING)
      if (EXISTS "/etc/debian_version") # is this a debian system ?
          if(CMAKE_LIBRARY_ARCHITECTURE)
              if("${CMAKE_INSTALL_PREFIX}" MATCHES "^/usr/?$")
                  set(_LIBDIR_DEFAULT "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
              endif()
              if(DEFINED _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX
                 AND "${_GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX}" MATCHES "^/usr/?$")
                  set(__LAST_LIBDIR_DEFAULT "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
              endif()
          endif()
      else() # not debian, rely on CMAKE_SIZEOF_VOID_P:
          if(NOT DEFINED CMAKE_SIZEOF_VOID_P)
              message(AUTHOR_WARNING
                  "Unable to determine default CMAKE_INSTALL_LIBDIR directory "
                  "because no target architecture is known. "
                  "Please enable at least one language before including GNUInstallDirs.")
          else()
              if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
                  set(_LIBDIR_DEFAULT "lib64")
                  if(DEFINED _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX)
                      set(__LAST_LIBDIR_DEFAULT "lib64")
                  endif()
              endif()
          endif()
      endif()
  endif()
endmacro()

#------------------------------------------------------------------------------#
# macro CACHE_VARIABLES_FOR_REFERENCE(...)
#           Provide a list of variables that will be stored in the cache
#               as {variable_name}_REF for later reference
#           This version does not force the cache to be updated
macro(CACHE_VARIABLES_FOR_REFERENCE)
    foreach(_var ${ARGN})
        set(${_var}_REF ${${_var}} CACHE STRING
            "Cached reference of ${_var} under ${_var}_REF for later comparison")
    endforeach()
endmacro()
#------------------------------------------------------------------------------#
# macro CACHE_VARIABLES_FOR_REFERENCE(...)
#           Provide a list of variables that will be stored in the cache
#              as {variable_name}_REF for later reference
#           This version forces the cache to be updated
macro(UPDATE_REFERENCE_CACHE_VARIABLES)
    foreach(_var ${ARGN})
        set(${_var}_REF ${${_var}} CACHE STRING
            "Cached reference of ${_var} under ${_var}_REF for later comparison" FORCE)
    endforeach()
endmacro()
#------------------------------------------------------------------------------#

cmake_policy(POP)

