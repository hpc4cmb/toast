# - Setup of general build options for  Libraries
#
# In addition to the core compiler/linker flags (configured in the
# MakeRules_<LANG>.cmake files), the build may require
# further configuration. This module performs this task whicj includes:
#
#  1) Extra build modes for developers
#  2) Additional compiler definitions to optimize
#     performance.
#  3) Additional compiler flags which may be added optionally.
#  4) Whether to build shared and/or static libraries.
#
#

INCLUDE(MacroUtilities)

#-----------------------------------------------------------------------
# Set up Build types or configurations
# If further tuning of compiler flags is needed then it should be done here.
# (It can't be done in the make rules override section).
# However, exercise care when doing this not to override existing flags!!
# We don't do this on WIN32 platforms yet because of some teething issues
# with compiler specifics and linker flags
IF(NOT WIN32)
  INCLUDE(BuildModes)
ENDIF()

#-----------------------------------------------------------------------
# Setup Shared and/or Static Library builds
# We name these options without a '${PROJECT_NAME}_' prefix because they are
# really higher level CMake options.
# Default to building shared libraries, mark options as advanced because
# most user should not have to touch them.

OPTION(BUILD_SHARED_LIBS "Build shared libraries" ON)
OPTION(BUILD_STATIC_LIBS "Build static libraries" OFF)

MARK_AS_ADVANCED(BUILD_SHARED_LIBS BUILD_STATIC_LIBS)

IF(BUILD_SHARED_LIBS AND NOT APPLE)
    SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
ENDIF()

IF(BUILD_STATIC_LIBS)
    OPTION(COMPILE_PIC "Compile position independent code" OFF)
    MARK_AS_ADVANCED(COMPILE_PIC)
    IF(COMPILE_PIC)
        SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
    ENDIF(COMPILE_PIC)
ENDIF(BUILD_STATIC_LIBS)

# Because both could be switched off accidently, FATAL_ERROR if neither
# option has been selected.
IF(NOT BUILD_STATIC_LIBS AND NOT BUILD_SHARED_LIBS)
    MESSAGE(WARNING "Neither static nor shared libraries will be built")
ENDIF()

foreach(_TYPE ARCHIVE LIBRARY RUNTIME)
    if(NOT UNIX)
        string(TOLOWER "${_TYPE}" _LTYPE)
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/outputs/${_LTYPE})
    else(NOT UNIX)
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    endif(NOT UNIX)
endforeach(_TYPE ARCHIVE LIBRARY RUNTIME)

