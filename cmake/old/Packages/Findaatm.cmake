#
# Module for locating aatm.
#
# Customizable variables:
#   aatm_ROOT
#       Specifies aatm's root directory.
#
#   aatm_64BIT_INTEGER
#       Enable aatm with 64-bit integer
#
#   aatm_THREADING
#       aatm threading model (options: sequential, OpenMP, TBB)
#
# Read-only variables:
#   aatm_FOUND
#       Indicates whether the library has been found.
#
#   aatm_INCLUDE_DIRS
#       Specifies aatm's include directory.
#
#   aatm_LIBRARIES
#       Specifies aatm libraries that should be passed to target_link_libararies.
#
#   aatm_<COMPONENT>_LIBRARIES
#       Specifies the libraries of a specific <COMPONENT>.
#
#   aatm_<COMPONENT>_FOUND
#       Indicates whether the specified <COMPONENT> was found.
#
#   aatm_CXX_LINK_FLAGS
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
# FITNESS FOR A PARTaatmLAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

INCLUDE (FindPackageHandleStandardArgs)


#----- aatm installation root
FIND_PATH (aatm_ROOT
  NAMES include/ATMVersion.h
  PATHS ${aatm_ROOT}
        ENV aatm_ROOT
        ENV aatmROOT
  DOC "aatm root directory")


#----- aatm include directory
FIND_PATH (aatm_INCLUDE_DIR
  NAMES ATMCommon.h
  HINTS ${aatm_ROOT}
        ENV aatm_ROOT
        ENV aatmROOT
  PATH_SUFFIXES include
  DOC "aatm include directory")


#----- aatm library
FIND_LIBRARY (aatm_LIBRARY
  NAMES aatm
  HINTS ${aatm_ROOT}
  PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR_DEFAULT} lib lib64
  DOC "aatm runtime library")


#----- Determine library's version
SET (_aatm_VERSION_HEADER ${aatm_INCLUDE_DIR}/ATMVersion.h)
IF (EXISTS ${_aatm_VERSION_HEADER})

    FILE (READ ${_aatm_VERSION_HEADER} _aatm_VERSION_CONTENTS)

    STRING (REGEX REPLACE ".*#define ATM_VERSION[ \t\"]+([0-9\.]+).*" "\\1"
        aatm_VERSION "${_aatm_VERSION_CONTENTS}")

    STRING (REPLACE "." ";" aatm_VERSION_LIST "${aatm_VERSION}")
    LIST(GET aatm_VERSION_LIST 0 aatm_VERSION_MAJOR)
    LIST(GET aatm_VERSION_LIST 1 aatm_VERSION_MINOR)
    LIST(GET aatm_VERSION_LIST 2 aatm_VERSION_PATCH)
    unset(aatm_VERSION_LIST)

    SET (aatm_VERSION_COMPONENTS 3)

ENDIF (EXISTS ${_aatm_VERSION_HEADER})

MARK_AS_ADVANCED (aatm_ROOT aatm_INCLUDE_DIR aatm_LIBRARY)

FIND_PACKAGE_HANDLE_STANDARD_ARGS (aatm REQUIRED_VARS aatm_ROOT
    aatm_INCLUDE_DIR aatm_LIBRARY VERSION_VAR aatm_VERSION)

SET (aatm_INCLUDE_DIRS ${aatm_INCLUDE_DIR})
SET (aatm_LIBRARIES ${aatm_LIBRARY})

