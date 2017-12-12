
cmake_policy(PUSH)
if(NOT CMAKE_VERSION VERSION_LESS 3.0)
    cmake_policy(SET CMP0043 NEW)
endif()

include(MacroLibraryTargets)

#------------------------------------------------------------------------------#

# cereal options
add_option(WITH_WERROR "Compile with '-Werror' C++ compiler flag" OFF NO_FEATURE)
add_option(THREAD_SAFE "Compile Cereal with THREAD_SAFE option" ON NO_FEATURE)
add_option(JUST_INSTALL_CEREAL "Skip testing of Cereal" ON NO_FEATURE)
add_option(SKIP_PORTABILITY_TEST "Skip Cereal portability test" ON NO_FEATURE)

set(DEV_WARNINGS ${CMAKE_SUPPRESS_DEVELOPER_WARNINGS})
# this gets annoying
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)

# add cereal
add_subdirectory(cereal)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ${DEV_WARNINGS} CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)

#------------------------------------------------------------------------------#

include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/src/cereal/include)
include_directories(${PROJECT_SOURCE_DIR}/src/libtoast)
include_directories(${PROJECT_SOURCE_DIR}/src/libtoast/capi)
include_directories(${PROJECT_SOURCE_DIR}/src/libtoast/fod)
include_directories(${PROJECT_SOURCE_DIR}/src/libtoast/atm)
include_directories(${PROJECT_SOURCE_DIR}/src/libtoast/test)
include_directories(${PROJECT_SOURCE_DIR}/src/libtoast/map)
include_directories(${PROJECT_SOURCE_DIR}/src/libtoast/test)
include_directories(${PROJECT_SOURCE_DIR}/src/libtoast/test/gtest)
include_directories(${PROJECT_SOURCE_DIR}/src/libtoast/tod)
include_directories(${PROJECT_SOURCE_DIR}/src/libtoast/math)
include_directories(${PROJECT_SOURCE_DIR}/src/libtoast/math/Random123)
include_directories(${PROJECT_SOURCE_DIR}/src/libtoast/math/Random123/conventional)
include_directories(${PROJECT_SOURCE_DIR}/src/libtoast/math/Random123/features)
include_directories(${PROJECT_SOURCE_DIR}/src/libtoast/util)
include_directories(${PROJECT_BINARY_DIR}/config)

add_subdirectory(libtoast)

#------------------------------------------------------------------------------#
#
set(PROJECT_LIBRARIES ${EXTERNAL_LIBRARIES})

# =============================================================================#
#   Library build
# =============================================================================#

#------------------------------------------------------------------------------#

if(BUILD_STATIC_LIBS)

    #--------------------------------------------------------------------------#
    # static toast library
    #--------------------------------------------------------------------------#
    add_library(${PROJECT_NAME}-static STATIC $<TARGET_OBJECTS:libtoast-object>)
    set_target_properties(${PROJECT_NAME}-static
        PROPERTIES OUTPUT_NAME      "toast"
                   VERSION          "${${PROJECT_NAME}_SHORT_VERSION}"
                   LINK_FLAGS       "${EXTERNAL_CXX_LINK_FLAGS}"
                   LINKER_LANGUAGE  CXX)
    target_link_libraries(${PROJECT_NAME}-static ${EXTERNAL_LIBRARIES})
    if(NOT BUILD_SHARED_LIBS)
        list(APPEND PROJECT_LIBRARIES ${PROJECT_NAME}-static)
    endif(NOT BUILD_SHARED_LIBS)
    set(PROJECT_LINK_TARGETS ${PROJECT_NAME}-static)
    #--------------------------------------------------------------------------#

endif()

#------------------------------------------------------------------------------#

if(BUILD_SHARED_LIBS)

    #--------------------------------------------------------------------------#
    # shared toast library
    #--------------------------------------------------------------------------#
    add_library(${PROJECT_NAME}-shared SHARED $<TARGET_OBJECTS:libtoast-object>)
    set_target_properties(${PROJECT_NAME}-shared
        PROPERTIES OUTPUT_NAME      "toast"
                   VERSION          "${${PROJECT_NAME}_SHORT_VERSION}"
                   SOVERSION        "${${PROJECT_NAME}_MAJOR_VERSION}"
                   LINK_FLAGS       "${EXTERNAL_CXX_LINK_FLAGS}"
                   )
    target_link_libraries(${PROJECT_NAME}-shared ${EXTERNAL_LIBRARIES})
    set(PROJECT_LINK_TARGETS ${PROJECT_NAME}-shared)
    list(APPEND PROJECT_LIBRARIES ${PROJECT_NAME}-shared)
    #--------------------------------------------------------------------------#

endif()


# =============================================================================#
#   Python build
# =============================================================================#
add_subdirectory(python)


# =============================================================================#
#   Executable build
# =============================================================================#
add_executable(toast_test libtoast/toast_test.cpp $<TARGET_OBJECTS:libtoast-object>)
target_link_libraries(toast_test
    ${PROJECT_LINK_TARGETS}
    ${EXTERNAL_LIBRARIES})


# =============================================================================#
#   Code coverage
# =============================================================================#
if(USE_COVERAGE)

    set(CTEST_MODEL "Continuous" CACHE STRING "Model for CTest")
    # CXX coverage
    SETUP_TARGET_FOR_COVERAGE(NAME cxx_toast_test_coverage
        EXECUTABLE toast_test
        DEPENDS toast_test)

    set(PYC_FILEPATH ${CMAKE_BINARY_DIR}/python
        CACHE PATH "Path to pyc_toast_test_coverage.py" FORCE)

    set(_PYFILES )
    foreach(_file ${PYTHON_INSTALL_FILES})
        set(_fname "${_file}")
        string(REPLACE "${CMAKE_SOURCE_DIR}/pipelines/" "toast." _fname "${_fname}")
        string(REPLACE "${CMAKE_CURRENT_LIST_DIR}/python/" "toast." _fname "${_fname}")
        string(REPLACE "${CMAKE_BINARY_DIR}/InstallTreeFiles/" "toast." _fname "${_fname}")
        string(REPLACE "/" "." _fname "${_fname}")
        configure_file(${_file} ${PYC_FILEPATH}/${_fname} COPYONLY)
    endforeach()

    # Python coverage
    FILE(WRITE ${CMAKE_BINARY_DIR}/pyc_toast_test_coverage.py
"#!/usr/bin/env python

import trace
import sys
import os
import toast

def run_tests():
    toast.test()

toast_path = os.path.abspath(toast.__file__)
print('toast path: {}'.format(toast_path))
# create a Trace object, telling it what to ignore, and whether to do tracing
# or line-counting or both.
tracer = trace.Trace(ignoredirs=[sys.exec_prefix], trace=0, count=1)
# run the new command using the given tracer
tracer.run('run_tests()')
# make a report, placing output in the current directory
r = tracer.results()
r.write_results(show_missing=False, coverdir='${PYC_FILEPATH}')

")

    SETUP_TARGET_FOR_COVERAGE(NAME pyc_toast_test_coverage
        EXECUTABLE ${PYTHON_EXECUTABLE} ${CMAKE_BINARY_DIR}/pyc_toast_test_coverage.py
        DEPENDS ${PROJECT_LINK_TARGETS})

endif(USE_COVERAGE)


# =============================================================================#
#   Export
get_directory_property(compile_defs COMPILE_DEFINITIONS)

include(GenerateConfigHeader)

set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)
set(PROJECT_SHARED_LIBRARY ${PROJECT_NAME}-shared)
set(PROJECT_STATIC_LIBRARY ${PROJECT_NAME}-static)
set(PACKAGE_VERSION "${${PROJECT_NAME}_VERSION}")
set(config_target ${PROJECT_NAME}LibraryDepends)
set(PROJECT_INCLUDE_DIRS_BUILD_TREE
    ${PROJECT_SOURCE_DIR}/src/libtoast)
foreach(_DIR capi fod atm map tod math math/Random123
        math/Random123/conventional math/Random123/features)
    list(APPEND PROJECT_INCLUDE_DIRS_BUILD_TREE ${PROJECT_SOURCE_DIR}/src/libtoast/${_DIR})
endforeach()
file(RELATIVE_PATH CMAKE_INSTALL_REL_INCLUDEDIR
    ${CMAKE_INSTALL_FULL_CMAKEDIR}
    ${CMAKE_INSTALL_FULL_INCLUDEDIR})
set(PROJECT_INCLUDE_DIRS_INSTALL_TREE
    \${${PROJECT_NAME}_CMAKE_DIR}/${CMAKE_INSTALL_REL_INCLUDEDIR})
set(_TARGETS )
set(PROJECT_LIBRARY_NAMES )

if(BUILD_SHARED_LIBS)
    list(APPEND _TARGETS ${PROJECT_NAME}-shared)
endif()
if(BUILD_STATIC_LIBS )
    list(APPEND _TARGETS ${PROJECT_NAME}-static)
endif()

list(GET _TARGETS 0 PROJECT_LIBRARY_NAMES)

# =============================================================================#
# BUILD TREE
# =============================================================================#
# Add all targets to the build-tree export set
export(TARGETS ${_TARGETS} FILE "${PROJECT_BINARY_DIR}/${PROJECT_NAME}LibraryDepends.cmake")
# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE ${PROJECT_NAME}LibraryDepends)
# Create the Config.cmake and ConfigHeader-${CMAKE_BUILD_TYPE}.cmake for the build tree
configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/ProjectExportConfig.cmake.in
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/ProjectExportConfigHeader.cmake.in
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigHeader-${CMAKE_BUILD_TYPE}.cmake"
    @ONLY)

# =============================================================================#
# INSTALL TREE
# =============================================================================#
# Create the Config.cmake and ConfigHeader-${CMAKE_BUILD_TYPE}.cmake for the install tree
configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/ProjectExportConfig.cmake.in
    "${PROJECT_BINARY_DIR}/InstallTreeFiles/${PROJECT_NAME}Config.cmake"
    @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/ProjectExportConfigHeader.cmake.in
    "${PROJECT_BINARY_DIR}/InstallTreeFiles/${PROJECT_NAME}ConfigHeader-${CMAKE_BUILD_TYPE}.cmake"
    @ONLY)

# Create the ConfigVersion.cmake for both
configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/ProjectExportConfigVersion.cmake.in
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" @ONLY)

# Install the ${PROJECT_NAME}Config*.cmake files
install(FILES
    "${PROJECT_BINARY_DIR}/InstallTreeFiles/${PROJECT_NAME}Config.cmake"
    "${PROJECT_BINARY_DIR}/InstallTreeFiles/${PROJECT_NAME}ConfigHeader-${CMAKE_BUILD_TYPE}.cmake"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
    COMPONENT dev)
# install the executables
install(TARGETS toast_test
    DESTINATION ${CMAKE_INSTALL_BINDIR})
# install the libraries
install(TARGETS ${_TARGETS}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    EXPORT ${PROJECT_NAME}LibraryDepends)
# Install the export set for use with the install-tree
install(EXPORT ${PROJECT_NAME}LibraryDepends
    DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
    COMPONENT dev)

# export the packages
export(PACKAGE ${PROJECT_NAME})

foreach(_cmake_file Compilers MacroUtilities)
    # Install CMake files
    install(FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/${_cmake_file}.cmake
        DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
        COMPONENT dev)
endforeach()

foreach(_cmake_file FindMKL FindSSE FindTBB)
    # Install CMake files
    install(FILES ${PROJECT_SOURCE_DIR}/cmake/Packages/${_cmake_file}.cmake
        DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
        COMPONENT dev)
endforeach()

cmake_policy(POP)
