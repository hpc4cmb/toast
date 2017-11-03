#
# Module for locating Elemental.
#
# Customizable variables:
#   Elemental_ROOT
#       Specifies Elemental's root directory.
#
# Read-only variables:
#   Elemental_FOUND
#       Indicates whether the library has been found.
#
#   Elemental_INCLUDE_DIRS
#       Specifies Elemental's include directory.
#
#   Elemental_LIBRARIES
#       Specifies Elemental libraries that should be passed to target_link_libararies.
#
#   Elemental_<COMPONENT>_LIBRARIES
#       Specifies the libraries of a specific <COMPONENT>.
#
#   Elemental_<COMPONENT>_FOUND
#       Indicates whether the specified <COMPONENT> was found.
#
#   Elemental_CXX_LINK_FLAGS
#       C++ linker compile flags
#
# Copyright (c) 2017 Jonathan Madsen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTElementalLAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

INCLUDE (FindPackageHandleStandardArgs)
set(Elemental_CMAKE_CONFIG ON CACHE BOOL "Use CMake configuration if found")
mark_as_advanced(Elemental_CMAKE_CONFIG)

#----- Elemental installation root
FIND_PATH (Elemental_ROOT
  NAMES include/El.hpp
  PATHS ${Elemental_ROOT}
        ENV Elemental_ROOT
        ENV ElementalROOT
  DOC "Elemental root directory")
mark_as_advanced(Elemental_ROOT)

if(Elemental_CMAKE_CONFIG AND
   EXISTS "${Elemental_ROOT}/CMake/elemental/ElementalConfig.cmake")

    include("${Elemental_ROOT}/CMake/elemental/ElementalConfig.cmake")

    FIND_PACKAGE_HANDLE_STANDARD_ARGS (Elemental REQUIRED_VARS Elemental_ROOT
        Elemental_INCLUDE_DIRS Elemental_LIBRARIES VERSION_VAR Elemental_VERSION)

else()
    #----- Elemental include directory
    FIND_PATH (Elemental_INCLUDE_DIR
        NAMES El.hpp
        HINTS ${Elemental_ROOT}
        PATH_SUFFIXES include
        DOC "Elemental include directory")


    #----- Elemental library
    FIND_LIBRARY (Elemental_LIBRARY
        NAMES El
        HINTS ${Elemental_ROOT}
        PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR_DEFAULT} lib lib64
        DOC "Elemental library")


    #----- Elemental library
    FIND_LIBRARY (Elemental_SuiteSparse_LIBRARY
        NAMES ElSuiteSparse
        HINTS ${Elemental_ROOT}
        PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR_DEFAULT} lib lib64
        DOC "Elemental SuiteSparse library")


    #----- Elemental library
    FIND_LIBRARY (Elemental_pmrrr_LIBRARY
        NAMES pmrrr
        HINTS ${Elemental_ROOT}
        PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR_DEFAULT} lib lib64
        DOC "Elemental pmrrr library")


    #----- Elemental library
    FIND_LIBRARY (Elemental_parmetis_LIBRARY
        NAMES parmetis
        HINTS ${Elemental_ROOT}
        PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR_DEFAULT} lib lib64
        DOC "Elemental parmetis library")


    #----- Elemental library
    FIND_LIBRARY (Elemental_metis_LIBRARY
        NAMES metis
        HINTS ${Elemental_ROOT}
        PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR_DEFAULT} lib lib64
        DOC "Elemental metis library")


    #----- Determine library's version
    SET (_Elemental_VERSION_HEADER ${Elemental_INCLUDE_DIR}/El/config.h)

    IF (EXISTS ${_Elemental_VERSION_HEADER})
    FILE (READ ${_Elemental_VERSION_HEADER} _Elemental_VERSION_CONTENTS)

    STRING (REPLACE "\"" "" _Elemental_VERSION_CONTENTS "${_Elemental_VERSION_CONTENTS}")
    STRING (REGEX REPLACE ".*#define EL_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1"
        Elemental_VERSION_MAJOR "${_Elemental_VERSION_CONTENTS}")
    STRING (REGEX REPLACE ".*#define EL_VERSION_MINOR[ \t]+([0-9]+).*" "\\1"
        Elemental_VERSION_MINOR "${_Elemental_VERSION_CONTENTS}")
    # -dev suffix will screw up CMake version check
    #STRING (REPLACE "-dev" "" Elemental_VERSION_MINOR "${Elemental_VERSION_MINOR}")

    SET (Elemental_VERSION ${Elemental_VERSION_MAJOR}.${Elemental_VERSION_MINOR})
    SET (Elemental_VERSION_COMPONENTS 2)
    ENDIF (EXISTS ${_Elemental_VERSION_HEADER})


    FIND_PACKAGE_HANDLE_STANDARD_ARGS (Elemental REQUIRED_VARS Elemental_ROOT
        Elemental_INCLUDE_DIR Elemental_LIBRARY VERSION_VAR Elemental_VERSION)

    SET (Elemental_INCLUDE_DIRS ${Elemental_INCLUDE_DIR})
    SET (Elemental_LIBRARIES ${Elemental_LIBRARY})
    FOREACH (_LIB SuiteSparse pmrrr parmetis metis)
        IF(Elemental_${_LIB}_LIBRARY)
            LIST (APPEND Elemental_LIBRARIES ${Elemental_${_LIB}_LIBRARY})
        ENDIF()
    ENDFOREACH()

endif()



#----- Elemental library
MARK_AS_ADVANCED(Elemental_INCLUDE_DIR
    Elemental_LIBRARY Elemental_SuiteSparse_LIBRARY
    Elemental_pmrrr_LIBRARY Elemental_parmetis_LIBRARY Elemental_metis_LIBRARY )
