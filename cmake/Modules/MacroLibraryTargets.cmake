# - Define useful macros
#
# A custom compile definition "DEVELOPER_<CONFIG>" is set on
# each target using the target property COMPILE_DEFINITIONS_<CONFIG>
# target property

if(__MACROLIBRARYTARGETS_ISLOADED)
  return()
endif()
set(__MACROLIBRARYTARGETS_ISLOADED TRUE)

include(CMakeParseArguments)
include(MacroDefineModule)

#-----------------------------------------------------------------------
# function compile_definitions_config(<target>)
#          Set a custom compile definition for a  target on a
#          per configuration basis:
#            For mode <CONFIG>, define DEVELOPER_<CONFIG>
#
function(compile_definitions_config _target)
    if(NOT TARGET ${_target})
        message(FATAL_ERROR "compile_definitions_config passed target '${_target}' which is not a valid CMake target")
    endif()

    if(CMAKE_CONFIGURATION_TYPES)
        # - Multimode tools
        foreach(_mode ${CMAKE_CONFIGURATION_TYPES})
            string(TOUPPER ${_mode} _mode_upper)
            set_property(TARGET ${_target}
                APPEND PROPERTY COMPILE_DEFINITIONS_${_mode_upper}
                DEVELOPER_${_mode_upper})
        endforeach()
    elseif(CMAKE_BUILD_TYPE)
        # - Single mode tools, only if set
        string(TOUPPER ${CMAKE_BUILD_TYPE} _mode_upper)
        set_property(TARGET ${_target}
            APPEND PROPERTY COMPILE_DEFINITIONS_${_mode_upper}
            DEVELOPER_${_mode_upper})
    endif()
endfunction()

#-----------------------------------------------------------------------
# - OBJECT_TARGET
# General build of a  library target
#
macro(OBJECT_TARGET)
    cmake_parse_arguments(OBJTARG
        ""
        "NAME"
        "SOURCES"
        ${ARGN}
    )

    # We have to distinguish the static from shared lib, so use -static in
    # name. Link its dependencies, and ensure we actually link to the
    # -static targets (We should strictly do this for the external
    # libraries as well if we want a pure static build).
    add_library(${OBJTARG_NAME} OBJECT ${OBJTARG_SOURCES})
    compile_definitions_config(${OBJTARG_NAME})
    target_compile_features(${OBJTARG_NAME} PUBLIC ${${PROJECT_NAME}_TARGET_COMPILE_FEATURES})

    # from DEFINE_MODULE
    get_property(SOURCE_GROUPS GLOBAL PROPERTY SOURCE_GROUPS)
    foreach(_src_grp ${SOURCE_GROUPS})
        get_property(HEADER_FOLDER GLOBAL PROPERTY ${_src_grp}_HEADER_FOLDER)
        get_property(SOURCE_FOLDER GLOBAL PROPERTY ${_src_grp}_SOURCE_FOLDER)
        get_property(HEADER_FILES GLOBAL PROPERTY ${_src_grp}_HEADER_FILES)
        get_property(SOURCE_FILES GLOBAL PROPERTY ${_src_grp}_SOURCE_FILES)
        source_group("${HEADER_FOLDER}" FILES ${HEADER_FILES})
        source_group("${SOURCE_FOLDER}" FILES ${SOURCE_FILES})
    endforeach()

    set_property(GLOBAL APPEND
      PROPERTY OBJECT_TARGETS ${OBJTARGET_NAME})

endmacro()


#-----------------------------------------------------------------------
# - GLOBAL_OBJECT_TARGET
# Build and install of a  category library
#
MACRO(GLOBAL_OBJECT_TARGET)
    CMAKE_PARSE_ARGUMENTS(GLOBOBJ
        "INSTALL_HEADERS"
        "NAME"
        "COMPONENTS"
        ${ARGN}
    )

    # We loop over the component sources one at a time,
    # appending properties as we go.
    foreach(_comp ${GLOBOBJ_COMPONENTS})
        include(${_comp})
        # In case we have a global lib with one component, ensure name gets set
        if(NOT GLOBOBJ_NAME)
            set(GLOBOBJ_NAME ${MODULE_NAME})
        endif()

        list(APPEND ${GLOBOBJ_NAME}_GLOBAL_SOURCES ${${MODULE_NAME}_SOURCES})
        list(APPEND ${GLOBOBJ_NAME}_GLOBAL_HEADERS ${${MODULE_NAME}_HEADERS})

        list(APPEND ${GLOBOBJ_NAME}_LINK_LIBRARIES ${${MODULE_NAME}_LINK_LIBRARIES})
        list(APPEND ${GLOBOBJ_NAME}_BUILDTREE_INCLUDES ${${MODULE_NAME}_INCDIR})
    endforeach()

    # Filter out duplicates in LINK_LIBRARIES
    if(${GLOBOBJ_NAME}_LINK_LIBRARIES)
        list(REMOVE_DUPLICATES ${GLOBOBJ_NAME}_LINK_LIBRARIES)
    endif()

    OBJECT_TARGET(NAME ${GLOBOBJ_NAME}
        SOURCES
            ${${GLOBOBJ_NAME}_GLOBAL_SOURCES}
            ${${GLOBOBJ_NAME}_GLOBAL_HEADERS}
    )

    set_property(GLOBAL APPEND
        PROPERTY GLOBAL_OBJECT_TARGETS \$<TARGET_OBJECTS:${GLOBOBJ_NAME}>)

    # Header install?
    if(GLOBOBJ_INSTALL_HEADERS)
        foreach(_header ${${GLOBOBJ_NAME}_GLOBAL_HEADERS})
            file(RELATIVE_PATH _fpath "${CMAKE_CURRENT_LIST_DIR}" "${_header}")
            get_filename_component(_fpath ${_fpath} PATH)
            install(FILES ${_header}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/${_fpath}
                COMPONENT Development)
        endforeach()
    endif()

    # Store the include path of the component so that the build tree
    # config file can pick up all needed header paths
    set_property(GLOBAL APPEND
                    PROPERTY BUILDTREE_INCLUDE_DIRS ${${GLOBOBJ_NAME}_BUILDTREE_INCLUDES})
    set_property(GLOBAL APPEND
                    PROPERTY INSTALL_HEADER_FILES ${${GLOBOBJNAME}_GLOBAL_HEADERS})

ENDMACRO()
