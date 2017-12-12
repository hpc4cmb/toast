# - Macros for organizing and specializing code in modules
#
# This file defines the following macros for  developers needing
# utlities for specializing source file properties
#
#
# It will also add the module include directory to the list of directories
# using include_directories
#
#

# - Include guard
if(__macrodefinemodule_isloaded)
  return()
endif()
set(__macrodefinemodule_isloaded YES)

include(CMakeParseArguments)


#-----------------------------------------------------------------------
# macro add_compile_flags(SOURCES <source1> ... <sourceN>
#                                      COMPILE_FLAGS <def1> ... <defN>
#                                      )
#       Add extra compile definitions to a specific list of sources.
#       Macroized to handle the need to specify absolute paths.
#
macro(add_compile_flags)
    cmake_parse_arguments(ADDDEF
        ""
        ""
        "SOURCES;COMPILE_FLAGS"
        ${ARGN}
    )

    # We assume that the sources have been added at the level of a
    # a sources.cmake, so are inside the src subdir of the sources.cmake
    get_filename_component(_ACD_BASE_PATH ${CMAKE_CURRENT_LIST_FILE} PATH)

    # Now for each file, add the definitions
    foreach(_acd_source ${ADDDEF_SOURCES})
        # add base path
        if(NOT IS_ABSOLUTE ${_acd_source})
            set(_acd_full_source ${_ACD_BASE_PATH}/${_acd_source})
        else()
            set(_acd_full_source ${_acd_source})
        endif()
        
        # Extract any existing compile definitions
        get_source_file_property(_acd_existing_properties
                                 ${_acd_full_source}
                                 COMPILE_FLAGS)

        if(_acd_existing_properties)
            set(_acd_new_defs ${_acd_existing_properties}
                              ${ADDDEF_COMPILE_FLAGS})
        else()
            set(_acd_new_defs ${ADDDEF_COMPILE_FLAGS})
        endif()

        # quote compile defs because this must epand to space separated list
        set_source_files_properties(${_acd_full_source}
                                    PROPERTIES COMPILE_FLAGS "${_acd_new_defs}")
    endforeach()
endmacro()


#-----------------------------------------------------------------------
# macro define_module(NAME <name>
#           HEADER_EXT <header_ext1> <header_ext2> ... <header_extN>
#           SOURCE_EXT <source_ext1> <source_ext2> ... <source_extN>
#           LINK_LIBRARIES <lib1> ... <lib2>)
#       Define a  Module's source extensions and what internal and external
#       libraries it links to.
#
macro(define_module)
    cmake_parse_arguments(DEFMOD
        "NO_SOURCE_GROUP"
        "NAME;HEADER_DIR;SOURCE_DIR;LANG"
        "HEADER_EXT;SOURCE_EXT;HEADERS;SOURCES;EXCLUDE;LINK_LIBRARIES;NOTIFY_EXCLUDE;COMPILE_FLAGS"
        ${ARGN}
        )

    set(_LANG CXX)
    if(DEFMOD_LANG)
        set(_LANG ${DEFMOD_LANG})
    endif()

    list(APPEND DEFMOD_EXCLUDE ${DEFMOD_NOTIFY_EXCLUDE})
    if(DEFMOD_EXCLUDE)
      list(REMOVE_DUPLICATES DEFMOD_EXCLUDE)
    endif()
    # set the filename
    set(MODULE_NAME ${DEFMOD_NAME})
    # get the path
    get_filename_component(${MODULE_NAME}_BASEDIR ${CMAKE_CURRENT_LIST_FILE} PATH)

    # Header directory
    if("${DEFMOD_HEADER_DIR}" STREQUAL "")
        set(${MODULE_NAME}_INCDIR ${${MODULE_NAME}_BASEDIR})
    else()
        set(${MODULE_NAME}_INCDIR "${${MODULE_NAME}_BASEDIR}/${DEFMOD_HEADER_DIR}")
    endif()
    # Source directory
    if("${DEFMOD_SOURCE_DIR}" STREQUAL "")
        set(${MODULE_NAME}_SRCDIR ${${MODULE_NAME}_BASEDIR})
    else()
        set(${MODULE_NAME}_SRCDIR "${${MODULE_NAME}_BASEDIR}/${DEFMOD_SOURCE_DIR}")
    endif()

    # list of files
    set(${MODULE_NAME}_HEADERS ${DEFMOD_HEADERS})
    set(${MODULE_NAME}_SOURCES ${DEFMOD_SOURCES})

    # GLOB the headers with provided extensions
    foreach(_ext ${DEFMOD_HEADER_EXT})
        file(GLOB _headers ${${MODULE_NAME}_INCDIR}/*${_ext})
        list(APPEND ${MODULE_NAME}_HEADERS ${_headers})
    endforeach()

    # GLOB the sources with provided extensions
    foreach(_ext ${DEFMOD_SOURCE_EXT})
        file(GLOB _sources ${${MODULE_NAME}_SRCDIR}/*${_ext})
        list(APPEND ${MODULE_NAME}_SOURCES ${_sources})
    endforeach()

    # Append the linked libraries
    foreach(_lib ${DEFMOD_LINK_LIBRARIES})
        list(APPEND ${MODULE_NAME}_LINK_LIBRARIES _lib)
    endforeach()

    set(_custom_header_list )
    set(_custom_source_list )
    # Check for files in custom target list
    foreach(_custom ${CUSTOM_TARGET_LIST})
        get_filename_component(_custom "${_custom}" NAME_WE)
        foreach(_ext ${DEFMOD_HEADER_EXT})
            set(_file "${${MODULE_NAME}_INCDIR}/${_custom}${_ext}")
            list(FIND ${MODULE_NAME}_HEADERS "${_file}" _found)
            if(NOT _found LESS 0)
                list(APPEND _custom_header_list ${_file})
                list(APPEND DEFMOD_EXCLUDE ${_custom})
            endif()
        endforeach()
        # loop over the source extensions
        foreach(_ext ${DEFMOD_SOURCE_EXT})
            set(_file "${${MODULE_NAME}_SRCDIR}/${_custom}${_ext}")
            list(FIND ${MODULE_NAME}_SOURCES "${_file}" _found)
            if(NOT _found LESS 0)
                list(APPEND _custom_source_list ${_file})
                list(APPEND DEFMOD_EXCLUDE ${_custom})
            endif()
        endforeach()
    endforeach()
    if(NOT "${_custom_header_list}" STREQUAL "")
        set_property(GLOBAL APPEND PROPERTY
                     CUSTOM_TARGET_HEADER_FILES ${_custom_header_list})
    endif()
    if(NOT "${_custom_source_list}" STREQUAL "")
        set_property(GLOBAL APPEND PROPERTY
                     CUSTOM_TARGET_SOURCE_FILES ${_custom_source_list})
    endif()

    # Remove the explicitly excluded files
    foreach(_ignore ${DEFMOD_EXCLUDE})
        # loop over the header extensions
        if(${MODULE_NAME}_HEADERS)
            foreach(_ext ${DEFMOD_HEADER_EXT})
                list(REMOVE_ITEM ${MODULE_NAME}_HEADERS
                    ${${MODULE_NAME}_INCDIR}/${_ignore}${_ext})
            endforeach()
        endif()
        # loop over the source extensions
        if(${MODULE_NAME}_SOURCES)
            foreach(_ext ${DEFMOD_SOURCE_EXT})
                list(REMOVE_ITEM ${MODULE_NAME}_SOURCES
                    ${${MODULE_NAME}_SRCDIR}/${_ignore}${_ext})
            endforeach()
        endif()
        list(FIND DEFMOD_NOTIFY_EXCLUDE ${_ignore} _notify)
        if(NOT _notify LESS 0)
           # Tell the user this file is not being compiled
           message(STATUS "${MODULE_NAME} module is not compiling : ${${MODULE_NAME}_BASEDIR}/${_ignore}")
         endif(NOT _notify LESS 0)
    endforeach()
    # include the directory
    include_directories(${${MODULE_NAME}_INCDIR})

    set_source_files_properties(${${MODULE_NAME}_SOURCES} PROPERTIES LANGUAGE ${_LANG})

    set(${MODULE_NAME}_LINK_LIBRARIES ${DEFMOD_LINK_LIBRARIES})

    # check for existance of modules.cmake and if it exists
    # add it to list of files to include
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/modules.cmake")
        set_property(GLOBAL APPEND PROPERTY MODULE_FILES
            ${CMAKE_CURRENT_LIST_DIR}/modules.cmake)
    endif()

    if(DEFMOD_COMPILE_FLAGS)
        add_compile_flags(SOURCES ${${MODULE_NAME}_HEADERS} ${${MODULE_NAME}_SOURCES}
            COMPILE_FLAGS ${DEFMOD_COMPILE_FLAGS})
    endif(DEFMOD_COMPILE_FLAGS)
        
    if(NOT ${DEFMOD_NO_SOURCE_GROUP})
        set_property(GLOBAL APPEND PROPERTY SOURCE_GROUPS ${MODULE_NAME})
        STRING(REPLACE "." "\\\\" _mod_name ${MODULE_NAME})
        set(_inc_folder "${_mod_name}\\Header Files")
        set(_imp_folder "${_mod_name}\\Source Files")
        set_property(GLOBAL PROPERTY ${MODULE_NAME}_HEADER_FOLDER ${_inc_folder})
        set_property(GLOBAL PROPERTY ${MODULE_NAME}_SOURCE_FOLDER ${_imp_folder})
        set_property(GLOBAL PROPERTY ${MODULE_NAME}_HEADER_FILES ${${MODULE_NAME}_HEADERS})
        set_property(GLOBAL PROPERTY ${MODULE_NAME}_SOURCE_FILES ${${MODULE_NAME}_SOURCES})
    endif()

endmacro()
