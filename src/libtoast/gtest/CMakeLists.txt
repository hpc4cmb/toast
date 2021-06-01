# NOTE:  Changed for consistency with the toast requirements
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

if (POLICY CMP0063)
  cmake_policy(SET CMP0063 NEW)
endif (POLICY CMP0063)

project( googletest-distribution )

enable_testing()

option(BUILD_GTEST "Builds the googletest subproject" OFF)

#Note that googlemock target already builds googletest
option(BUILD_GMOCK "Builds the googlemock subproject" ON)

if(BUILD_GMOCK)
  add_subdirectory( googlemock )
elseif(BUILD_GTEST)
  add_subdirectory( googletest )
endif()
