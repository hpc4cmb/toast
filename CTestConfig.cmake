#=============================================================================
# CMake - Cross Platform Makefile Generator
# Copyright 2000-2009 Kitware, Inc., Insight Software Consortium
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================

find_program(CTEST_HOSTNAME_COMMAND NAMES hostname)
execute_process(COMMAND ${CTEST_HOSTNAME_COMMAND} 
    OUTPUT_VARIABLE HOSTNAME WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_STRIP_TRAILING_WHITESPACE)
    
set(CTEST_SITE                  "${HOSTNAME}")
set(CTEST_PROJECT_NAME          "TOAST")
set(CTEST_NIGHTLY_START_TIME    "00:00:00 PDT")
set(CTEST_DROP_METHOD           "http")
set(CTEST_DROP_SITE             "jonathan-madsen.info")
set(CTEST_DROP_LOCATION         "/cdash/public/submit.php?project=TOAST")
set(CTEST_DROP_SITE_CDASH       TRUE)
set(CTEST_CDASH_VERSION         "1.6")
set(CTEST_CDASH_QUERY_VERSION   TRUE)
