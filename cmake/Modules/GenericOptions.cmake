

# - Include guard
#if(__genericcmakeoptions_isloaded)
#  return()
#endif()
#set(__genericcmakeoptions_isloaded YES)

include(MacroUtilities)

################################################################################
# Add option + feature
################################################################################
MACRO(ADD_OPTION_AND_FEATURE _NAME _MESSAGE _DEFAULT _FEATURE)
    OPTION(${_NAME} "${_MESSAGE}" ${_DEFAULT})
    IF(NOT "${_FEATURE}" STREQUAL "NO_FEATURE")
        ADD_FEATURE(${_NAME} "${_MESSAGE}")
    ELSE()
        MARK_AS_ADVANCED(${_NAME})
    ENDIF()
ENDMACRO()


################################################################################
# RPATH Relink Options
################################################################################

MACRO(OPTION_SET_BUILD_RPATH DEFAULT)
    set(_message "Set the build RPATH to local directories, relink to install")
    set(_message "${_message} directories at install time")
    ADD_OPTION_AND_FEATURE(SET_BUILD_RPATH "${_message}" ${DEFAULT} "${ARGN}")

    IF(SET_BUILD_RPATH)
        if(APPLE)
            set(CMAKE_MACOSX_RPATH 1)
        endif(APPLE)

        # the RPATH to be used when installing
        if(DEFINED LIBRARY_INSTALL_DIR)
            SET(CMAKE_INSTALL_RPATH "${LIBRARY_INSTALL_DIR}")
        else()
            SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
        endif()

        # use, i.e. don't skip the full RPATH for the build tree
        SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

        # when building, don't use the install RPATH already
        # (but later on when installing)
        SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

        # add the automatically determined parts of the RPATH
        # which point to directories outside the build tree to the install RPATH
        SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

        # the RPATH to be used when installing, but only if it's not a system directory
        LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
        IF("${isSystemDir}" STREQUAL "-1")
           SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
        ENDIF("${isSystemDir}" STREQUAL "-1")

    ENDIF(SET_BUILD_RPATH)
    unset(_message)
ENDMACRO(OPTION_SET_BUILD_RPATH)


################################################################################
# Turn on GProf based profiling
################################################################################

MACRO(OPTION_GPROF DEFAULT)
    IF(CMAKE_COMPILER_IS_GNUCXX)
        ADD_OPTION_AND_FEATURE(ENABLE_GPROF "Compile using -g -pg for gprof output" ${DEFAULT} "${ARGN}")
        IF(ENABLE_GPROF)
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pg")
            SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -pg")
            SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -pg")
        ENDIF(ENABLE_GPROF)
    ENDIF(CMAKE_COMPILER_IS_GNUCXX)
ENDMACRO(OPTION_GPROF)


################################################################################
# Look for accelerate, if found, add proper includes
################################################################################

MACRO(OPTION_ACCELERATE_FRAMEWORK DEFAULT)
    IF(APPLE)
        set(_message "Use Accelerate Framework for Math (Adds -D_ACCELERATE_ for compiling and Accelerate Framework linking)")
        ADD_OPTION_AND_FEATURE(ACCELERATE_FRAMEWORK ${_message} ${DEFAULT} "${ARGN}")
        IF(ACCELERATE_FRAMEWORK)
            FIND_LIBRARY(ACCELERATE_LIBRARY Accelerate REQUIRED
                PATHS "/System/Library/Frameworks/Accelerate.framework/" )
            FIND_PATH(ACCELERATE_INCLUDE_DIR Accelerate.h REQUIRED
                PATHS "/System/Library/Frameworks/Accelerate.framework/Headers/ ")
            set(ACCELERATE_FOUND OFF)
            if(ACCELERATE_LIBRARY AND ACCELERATE_INCLUDE_DIR)
                add_definitions(-DUSE_ACCELERATE)
                set(ACCELERATE_INCLUDE_DIRS ${ACCELERATE_INCLUDE_DIR})
                set(ACCELERATE_LIBRARIES ${ACCELERATE_LIBRARY})
                set(ACCELERATE_FOUND ON)
            endif()
        ENDIF(ACCELERATE_FRAMEWORK)
        unset(_message)
    ENDIF(APPLE)
ENDMACRO(OPTION_ACCELERATE_FRAMEWORK)

