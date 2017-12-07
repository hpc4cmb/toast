#
# - MakeRulesCXX
# Sets the default make rules for a CXX build, specifically the
# initialization of the compiler flags on a platform and compiler
# dependent basis
#
# Settings for each compiler are handled in a dedicated function.
# Whilst we only handle GNU, Clang and Intel, these are sufficiently
# different in the required flags that individual handling is needed.
#

include(Compilers)

#------------------------------------------------------------------------------#
# MSVC
#------------------------------------------------------------------------------#
if(MSVC)
    message(FATAL_ERROR "${PROJECT_NAME} does not support Windows")
endif()

#------------------------------------------------------------------------------#
# macro for finding the x86intrin.h header path that needs to be explicitly
# in the search path
#   - Otherwise, icc/icpc will use the GNU x86intrin.h header and
#     this header causes quite a bit of compilation problems
# - The Intel compiler has alot of links and not an easy structure to search
#   so this macro pretty much assumes the x86intrinc.h file will be in
#   an icc/ folder although (icpc/ and icl/) are also permitted)
# - When explicitly setting this folder, be sure to include icc/ or whatever
#   the parent directory is
#------------------------------------------------------------------------------#
macro(add_intel_intrinsic_include_dir)

    find_program(ICC_COMPILER icc)
    if(NOT ICC_COMPILER)
        set(ICC_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH 
            "Path to CMake compiler (should be icc)" FORCE) 
    endif(NOT ICC_COMPILER)

    get_filename_component(COMPILER_DIR "${ICC_COMPILER}" REALPATH)
    set(SYSNAME "${CMAKE_SYSTEM_NAME}")
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" LSYSNAME)
    if(APPLE)
        set(LSYSNAME "mac")
    endif(APPLE)

    #--------------------------------------------------------------------------#
    find_path(INTEL_ICC_CPATH
        NAMES x86intrin.h
        HINTS
            ENV INTEL_ICC_CPATH
            ${COMPILER_DIR}
            ${COMPILER_DIR}/..
            ${COMPILER_DIR}/../..
            /opt/intel/compilers_and_libraries
            /opt/intel/compilers_and_libraries/${SYSNAME}
            /opt/intel/compilers_and_libraries/${LSYSNAME}
            ENV INTEL_ROOT
        PATHS
            ENV CPATH
            ENV INTEL_ICC_CPATH
            ${COMPILER_DIR}
            ${COMPILER_DIR}/..
            ${COMPILER_DIR}/../..
            /opt/intel/compilers_and_libraries
            /opt/intel/compilers_and_libraries/${SYSNAME}
            /opt/intel/compilers_and_libraries/${LSYSNAME}
            ENV INTEL_ROOT
        PATH_SUFFIXES
            icc
            include/icc
            ${SYSNAME}/include/icc
            ${SYSNAME}/compiler/include/icc
            ${LSYSNAME}/compiler/include/icc
        NO_SYSTEM_ENVIRONMENT_PATH
        DOC "Include path for the ICC Compiler (need to correctly compile intrinsics)")
    #--------------------------------------------------------------------------#

    #--------------------------------------------------------------------------#
    if(NOT EXISTS "${INTEL_ICC_CPATH}/x86intrin.h")
        set(_expath "/opt/intel/compilers_and_libraries/${LSYSNAME}/compiler/include/icc")
        set(_msg "INTEL_ICC_CPATH was not found! Please specify the path to \"x86intrin.h\"")
        add(_msg "in the Intel ICC compiler path. Using the GNU header for this file")
        add(_msg "typically results in a failure to compile.\nExample:\n")
        add(_msg "\t-DINTEL_ICC_CPATH=${_expath}")
        add(_msg "\nfor \"${_expath}/x86intrin.h\"\n")
        message(WARNING "${_msg}")
    endif()
    #--------------------------------------------------------------------------#

    if(INTEL_ICC_CPATH)
        include_directories(${INTEL_ICC_CPATH})
    endif(INTEL_ICC_CPATH)

endmacro()

#------------------------------------------------------------------------------#
# GNU C++ or LLVM/Clang Compiler on all(?) platforms
#
if(CMAKE_CXX_COMPILER_IS_GNU OR CMAKE_CXX_COMPILER_IS_CLANG)

    add(_std_flags   "-Wno-deprecated $ENV{CXX_FLAGS}")
    add(_loud_flags  "-Wwrite-strings -Wpointer-arith -Woverloaded-virtual")
    add(_loud_flags  "-Wshadow -Wextra -pedantic")
    add(_quiet_flags "-Wno-unused-function -Wno-unused-variable")
    add(_quiet_flags "-Wno-attributes -Wno-expansion-to-defined")
    add(_quiet_flags "-Wno-unused-parameter -Wno-sign-compare")
    add(_quiet_flags "-Wno-attributes -Wno-expansion-to-defined")
    add(_quiet_flags "-Wno-type-limits -Wno-terminate -Wno-div-by-zero")
    add(_quiet_flags "-Wno-maybe-uninitialized -Wno-missing-field-initializers")
    add(_quiet_flags "-Wno-unused-but-set-variable -Wno-unused-local-typedefs")
    add(_quiet_flags "-Wno-implicit-fallthrough")
    add(_fast_flags  "-ftree-vectorize -ftree-loop-vectorize")
    add(_std_flags   "-faligned-new -fdiagnostics-color=always")
    add(_std_flags   "-Qunused-arguments")

    if(NOT CMAKE_CXX_COMPILER_IS_GNU)
        INCLUDE_DIRECTORIES("/usr/include/libcxxabi")
    endif()

    option(ENABLE_GCOV "Enable compilation flags for GNU coverage tool (gcov)" OFF)
    mark_as_advanced(ENABLE_GCOV)
    if(ENABLE_GCOV)
        add(_std_flags "-fprofile-arcs -ftest-coverage")
    endif(ENABLE_GCOV)

    add_c_flags(_std_flags "${_std_flags}")
    add_c_flags(_par_flags "${_par_flags}")
    add_c_flags(_loud_flags "${_loud_flags}")
    add_c_flags(_quiet_flags "${_quiet_flags}")
    add_c_flags(_fast_flags "${_fast_flags}")

    set_no_duplicates(CMAKE_CXX_FLAGS_INIT                "-W -Wall ${_std_flags}")
    set_no_duplicates(CMAKE_CXX_FLAGS_DEBUG_INIT          "-g -DDEBUG ${_loud_flags}")
    set_no_duplicates(CMAKE_CXX_FLAGS_MINSIZEREL_INIT     "-Os -DNDEBUG ${_quiet_flags}")
    set_no_duplicates(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-g -O2 ${_fast_flags}")
    set_no_duplicates(CMAKE_CXX_FLAGS_RELEASE_INIT        "-O3 -DNDEBUG ${_fast_flags} ${_quiet_flags}")

#------------------------------------------------------------------------------#
# Intel C++ Compilers
#
elseif(CMAKE_CXX_COMPILER_IS_INTEL)

    add(_std_flags   "-Wno-deprecated $ENV{CXX_FLAGS}")
    add(_loud_flags  "-Wwrite-strings -Wpointer-arith -Woverloaded-virtual")
    add(_loud_flags  "-Wshadow -Wextra -pedantic")
    add(_quiet_flags "-Wno-unused-function -Wno-unused-variable")
    add(_quiet_flags "-Wno-attributes -Wno-unused-but-set-variable")
    add(_quiet_flags "-Wno-unused-parameter -Wno-sign-compare")
    add(_std_flags   "-Wno-unknown-pragmas -Wno-deprecated")
    add(_extra_flags "-Wno-non-virtual-dtor -Wpointer-arith -Wwrite-strings -fp-model precise")
    add(_par_flags   "-parallel-source-info=2")

    add_intel_intrinsic_include_dir()

    add_cxx_flags(_std_flags "${_std_flags}")
    add_cxx_flags(_extra_flags "${_extra_flags}")
    add_cxx_flags(_par_flags "${_par_flags}")
    add_cxx_flags(_loud_flags "${_loud_flags}")
    add_cxx_flags(_quiet_flags "${_quiet_flags}")

    set_no_duplicates(CMAKE_CXX_FLAGS_INIT                "${_std_flags} ${_extra_flags}")
    set_no_duplicates(CMAKE_CXX_FLAGS_DEBUG_INIT          "-debug -DDEBUG ${_par_flags} ${_loud_flags}")
    set_no_duplicates(CMAKE_CXX_FLAGS_MINSIZEREL_INIT     "-Os -DNDEBUG ${_quiet_flags}")
    set_no_duplicates(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-O2 -debug ${_par_flags}")
    set_no_duplicates(CMAKE_CXX_FLAGS_RELEASE_INIT        "-Ofast -DNDEBUG ${_quiet_flags}")

#-----------------------------------------------------------------------
# IBM xlC compiler
#
elseif(CMAKE_CXX_COMPILER_IS_XLC)

    add_cxx_flags(CMAKE_CXX_FLAGS_INIT                "$ENV{CXX_FLAGS}")
    add_cxx_flags(CMAKE_CXX_FLAGS_DEBUG_INIT          "-g -qdbextra -qcheck=all -qfullpath -qtwolink -+")
    add_cxx_flags(CMAKE_CXX_FLAGS_MINSIZEREL_INIT     "-O2 -qtwolink -+")
    add_cxx_flags(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-O2 -g -qdbextra -qcheck=all -qfullpath -qtwolink -+")
    add_cxx_flags(CMAKE_CXX_FLAGS_RELEASE_INIT        "-O2 -qtwolink -+")

#---------------------------------------------------------------------
# HP aC++ Compiler
#
elseif(CMAKE_CXX_COMPILER_IS_HP_ACC)

    add_cxx_flags(CMAKE_CXX_FLAGS_INIT                "+DAportable +W823 $ENV{CXX_FLAGS}")
    add_cxx_flags(CMAKE_CXX_FLAGS_DEBUG_INIT          "-g")
    add_cxx_flags(CMAKE_CXX_FLAGS_MINSIZEREL_INIT     "-O3 +Onolimit")
    add_cxx_flags(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-O3 +Onolimit -g")
    add_cxx_flags(CMAKE_CXX_FLAGS_RELEASE_INIT        "+O2 +Onolimit")

#---------------------------------------------------------------------
# IRIX MIPSpro CC Compiler
#
elseif(CMAKE_CXX_COMPILER_IS_MIPS)

    add_cxx_flags(CMAKE_CXX_FLAGS_INIT                "-ptused -DSOCKET_IRIX_SOLARIS")
    add_cxx_flags(CMAKE_CXX_FLAGS_DEBUG_INIT          "-g")
    add_cxx_flags(CMAKE_CXX_FLAGS_MINSIZEREL_INIT     "-O -OPT:Olimit=5000")
    add_cxx_flags(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-O -OPT:Olimit=5000 -g")
    add_cxx_flags(CMAKE_CXX_FLAGS_RELEASE_INIT        "-O -OPT:Olimit=5000")

endif()


foreach(_init DEBUG RELEASE MINSIZEREL RELWITHDEBINFO)
    if(NOT "${CMAKE_CXX_FLAGS_${_init}_INIT}" STREQUAL "")
        string(REPLACE "  " " " CMAKE_CXX_FLAGS_${_init}_INIT "${CMAKE_CXX_FLAGS_${_init}_INIT}")
    endif()
endforeach()


