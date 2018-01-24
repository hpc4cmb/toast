# - Add ${CMAKE_PROJECT_NAME} specific build modes and additional flags
#
# This follows the guide on adding a new mode on the CMake wiki:
#
# http://www.cmake.org/Wiki/CMake_FAQ#How_can_I_extend_the_build_modes_with_a_custom_made_one_.3F
#
# ${CMAKE_PROJECT_NAME} supports the standard CMake build types/configurations of
#
# Release Debug MinSizeRel RelWithDebInfo
#
# In addition, two types specifically for development are added:
#
# TestRelease:
#   For trial production and extended testing. It has verbose
#   output, has debugging symbols, and adds definitions to allow FPE
#   and physics conservation law testing where supported.
#
# Maintainer:
#   For development of the toolkit. It adds debugging, and enables the use
#   of library specific debugging via standardized definitions.
#
# Compiler flags specific to these build types are set in the cache, and
# the  types are added to the CMAKE_BUILD_TYPE cache string and to
# CMAKE_CONFIGURATION_TYPES if appropriate to the build tool being used.
#


#-----------------------------------------------------------------------
# Update build type information ONLY for single mode build tools, adding
# default type if none has been set, but otherwise leaving value alone.
# NB: this doesn't allow "None" for the build type - would need something
# more sophiticated using an internal cache variable.
#
if(NOT CMAKE_BUILD_TYPE)
    # Default to a Release build if nothing else...
    set(CMAKE_BUILD_TYPE Release
      CACHE STRING "Choose the type of build, options are: Release MinSizeRel Debug RelWithDebInfo MinSizeRel."
      FORCE
      )
else()
    # Force to the cache, but use existing value.
    set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
      CACHE STRING "Choose the type of build, options are: Release MinSizeRel Debug RelWithDebInfo MinSizeRel."
      FORCE
      )
endif()

#-----------------------------------------------------------------------
# Update build flags for each build type
#
set(CONFIG_TYPES Debug Release RelWithDebInfo MinSizeRel
    CACHE STRING "Configuration types")
mark_as_advanced(CONFIG_TYPES)
foreach(type ${CONFIG_TYPES})
    string(TOUPPER "${type}" UTYPE)
    foreach(LANG C CXX)
        unset(CMAKE_${LANG}_FLAGS_${UTYPE} CACHE)
        set(CMAKE_${LANG}_FLAGS_${UTYPE} "${CMAKE_${LANG}_FLAGS_${UTYPE}_INIT} ${CMAKE_${LANG}_FLAGS_EXTRA}")
        string(REPLACE "  " " " CMAKE_${LANG}_FLAGS_${UTYPE} "${CMAKE_${LANG}_FLAGS_${UTYPE}}")
        string(REPLACE "  " " " CMAKE_${LANG}_FLAGS_${UTYPE} "${CMAKE_${LANG}_FLAGS_${UTYPE}}")
    endforeach(LANG C CXX)
endforeach()

foreach(LANG C CXX)
    unset(CMAKE_${LANG}_FLAGS CACHE)
    set(CMAKE_${LANG}_FLAGS "${CMAKE_${LANG}_FLAGS_INIT}")
endforeach(LANG C CXX)

