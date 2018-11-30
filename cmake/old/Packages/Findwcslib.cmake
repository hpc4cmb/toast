#
# Module for locating wcslib.
#
# Customizable variables:
#   wcslib_ROOT
#       Specifies wcslib's root directory.
#
#   wcslib_64BIT_INTEGER
#       Enable wcslib with 64-bit integer
#
#   wcslib_THREADING
#       wcslib threading model (options: sequential, OpenMP, TBB)
#
# Read-only variables:
#   wcslib_FOUND
#       Indicates whether the library has been found.
#
#   wcslib_INCLUDE_DIRS
#       Specifies wcslib's include directory.
#
#   wcslib_LIBRARIES
#       Specifies wcslib libraries that should be passed to target_link_libararies.
#
#   wcslib_<COMPONENT>_LIBRARIES
#       Specifies the libraries of a specific <COMPONENT>.
#
#   wcslib_<COMPONENT>_FOUND
#       Indicates whether the specified <COMPONENT> was found.
#
#   wcslib_CXX_LINK_FLAGS
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
# FITNESS FOR A PARTwcslibLAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

INCLUDE (FindPackageHandleStandardArgs)

set(INCLUDE_VERSION ${PACKAGE_FIND_VERSION_MAJOR}.${PACKAGE_FIND_VERSION_MINOR})

#----- wcslib installation root
FIND_PATH (wcslib_ROOT
  NAMES include/wcslib/wcs.h 
        include/wcslib-${INCLUDE_VERSION}/wcs.h
  PATHS ${wcslib_ROOT}
        ENV wcslib_ROOT
        ENV wcslibROOT
  DOC "wcslib root directory")


#----- wcslib include directory
FIND_PATH (wcslib_INCLUDE_DIR
  NAMES wcslib/wcsconfig.h
        wcslib-${INCLUDE_VERSION}/wcsconfig.h
  HINTS ${wcslib_ROOT}
        ENV wcslib_ROOT
        ENV wcslibROOT
  PATH_SUFFIXES include
  DOC "wcslib include directory")


#----- wcslib library
FIND_LIBRARY (wcslib_LIBRARY
  NAMES wcs
  HINTS ${wcslib_ROOT}
  PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR_DEFAULT} lib lib64
  DOC "wcslib runtime library")


#----- Determine library's version
FOREACH(_wcslib_VERSION_HEADER
    ${wcslib_INCLUDE_DIR}/wcslib/wcsconfig.h
    ${wcslib_INCLUDE_DIR}/wcslib-${INCLUDE_VERSION}/wcsconfig.h)
    
    IF (EXISTS ${_wcslib_VERSION_HEADER})
    
        FILE (READ ${_wcslib_VERSION_HEADER} _wcslib_VERSION_CONTENTS)
    
        STRING (REGEX REPLACE ".*#define WCSLIB_VERSION[ \t\"]+([0-9\.]+).*" "\\1"
            wcslib_VERSION "${_wcslib_VERSION_CONTENTS}")
    
        STRING (REPLACE "." ";" wcslib_VERSION_LIST "${wcslib_VERSION}")
        LIST(GET wcslib_VERSION_LIST 0 wcslib_VERSION_MAJOR)
        LIST(GET wcslib_VERSION_LIST 1 wcslib_VERSION_MINOR)
        unset(wcslib_VERSION_LIST)
    
        SET (wcslib_VERSION_COMPONENTS 2)
        
        BREAK()
        
    ENDIF (EXISTS ${_wcslib_VERSION_HEADER})

ENDFOREACH(_wcslib_VERSION_HEADER
    ${wcslib_INCLUDE_DIR}/wcslib/wcsconfig.h
    ${wcslib_INCLUDE_DIR}/wcslib-${INCLUDE_VERSION}/wcsconfig.h)


LIST(APPEND wcslib_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
MARK_AS_ADVANCED (wcslib_ROOT wcslib_INCLUDE_DIR wcslib_LIBRARY wcslib_BINARY_DIR)

FIND_PACKAGE_HANDLE_STANDARD_ARGS (wcslib REQUIRED_VARS wcslib_ROOT
    wcslib_INCLUDE_DIR wcslib_LIBRARY VERSION_VAR wcslib_VERSION)

SET (wcslib_INCLUDE_DIRS ${wcslib_INCLUDE_DIR})
SET (wcslib_LIBRARIES ${wcslib_LIBRARY})

