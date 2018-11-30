#
# Module for locating GCC OpenMP library (gomp).
#
# Customizable variables:
#   GOMP_ROOT
#       Specifies GOMP's root directory.
#
# Read-only variables:
#   GOMP_FOUND
#       Indicates whether the library has been found.
#
#   GOMP_INCLUDE_DIRS
#       Specifies GOMP's include directory.
#
#   GOMP_LIBRARIES
#       Specifies GOMP libraries that should be passed to target_link_libararies.
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
# FITNESS FOR A PARTGOMPLAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

INCLUDE (FindPackageHandleStandardArgs)
INCLUDE (GenericCMakeFunctions)

#----- GOMP suffixes
#----- works for Ubuntu. Not sure about other systems
GET_VERSION_COMPONENTS(GCC ${CMAKE_C_COMPILER_VERSION})
GET_VERSION_COMPONENTS(GXX ${CMAKE_CXX_COMPILER_VERSION})
STRING(TOLOWER "${CMAKE_SYSTEM_NAME}" SYS_NAME)
SET(GOMP_SUFFIXES
    ${CMAKE_SYSTEM_PROCESSOR}-${SYS_NAME}-gnu
    ${CMAKE_SYSTEM_PROCESSOR}-${SYS_NAME}-gnu/${GXX_MAJOR_VERSION}.${GXX_MINOR_VERSION}
    ${CMAKE_SYSTEM_PROCESSOR}-${SYS_NAME}-gnu/${GCC_MAJOR_VERSION}.${GCC_MINOR_VERSION}
    ${CMAKE_SYSTEM_PROCESSOR}-${SYS_NAME}-gnu/${GXX_MAJOR_VERSION}
    ${CMAKE_SYSTEM_PROCESSOR}-${SYS_NAME}-gnu/${GCC_MAJOR_VERSION}
    gcc/${CMAKE_SYSTEM_PROCESSOR}-${SYS_NAME}-gnu
    gcc/${CMAKE_SYSTEM_PROCESSOR}-${SYS_NAME}-gnu/${GXX_MAJOR_VERSION}.${GXX_MINOR_VERSION}
    gcc/${CMAKE_SYSTEM_PROCESSOR}-${SYS_NAME}-gnu/${GCC_MAJOR_VERSION}.${GCC_MINOR_VERSION}
    gcc/${CMAKE_SYSTEM_PROCESSOR}-${SYS_NAME}-gnu/${GXX_MAJOR_VERSION}
    gcc/${CMAKE_SYSTEM_PROCESSOR}-${SYS_NAME}-gnu/${GCC_MAJOR_VERSION})


#----- GOMP installation root
FIND_PATH (GOMP_ROOT
    NAMES include/omp.h
    HINTS
        ${GOMP_ROOT}
        ENV GOMP_ROOT
        /usr/lib64
        /usr/lib
        /usr/lib32
    PATH_SUFFIXES
        ${GOMP_SUFFIXES}
    DOC "GOMP root directory")
MARK_AS_ADVANCED(GOMP_ROOT)


#----- GOMP include directory
FIND_PATH (GOMP_INCLUDE_DIR
  NAMES omp.h
  HINTS ${GOMP_ROOT}
  PATH_SUFFIXES include
  DOC "GOMP include directory")


#----- GOMP library
FIND_LIBRARY (GOMP_LIBRARY
    NAMES gomp libgomp
    HINTS
        ENV LD_LIBRARY_PATH
        ENV LIBRARY_PATH
        ENV DYLD_LIBRARY_PATH
    PATH_SUFFIXES ${GOMP_SUFFIXES}
    DOC "GCC OpenMP library")


#----- assume the compiler can find it
IF(NOT GOMP_LIBRARY)
    SET(GOMP_LIBRARY gomp)
ENDIF(NOT GOMP_LIBRARY)


#----- Standard arguments
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GOMP REQUIRED_VARS GOMP_LIBRARY)


#----- FindGOMP non-cache variables
SET(GOMP_LIBRARIES ${GOMP_LIBRARY})

IF(GOMP_INCLUDE_DIR)
    SET (GOMP_INCLUDE_DIRS ${GOMP_INCLUDE_DIR})
ENDIF(GOMP_INCLUDE_DIR)


#----- Hide in advanced section of GUI
MARK_AS_ADVANCED(GOMP_ROOT GOMP_INCLUDE_DIR GOMP_LIBRARY)
