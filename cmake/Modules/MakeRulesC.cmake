#
# - MakeRulesC
# Sets the default make rules for a C build, specifically the
# initialization of the compiler flags on a platform and compiler
# dependent basis
#
# Settings for each compiler are handled in a dedicated function.
# Whilst we only handle GNU, Clang and Intel, these are sufficiently
# different in the required flags that individual handling is needed.
#

include(Compilers)

#------------------------------------------------------------------------------#
# macro for adding flags to variable
#------------------------------------------------------------------------------#
macro(add _VAR _FLAG)
    set(${_VAR} "${${_VAR}} ${_FLAG}")
endmacro()

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
macro(get_intel_intrinsic_include_dir)

    get_filename_component(COMPILER_DIR "${CMAKE_C_COMPILER}" PATH)
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

endmacro()

#------------------------------------------------------------------------------#
# GNU C++ or LLVM/Clang Compiler on all(?) platforms
#
if(CMAKE_C_COMPILER_IS_GNU OR CMAKE_C_COMPILER_IS_CLANG)

    set(_def_cxx     "-Wno-deprecated $ENV{C_FLAGS} -Wno-missing-field-initializers")
    add(_def_cxx     "-Wno-unused-parameter -Wno-sign-compare")
    set(_verb_flags  "-Wwrite-strings -Wpointer-arith -Woverloaded-virtual")
    add(_verb_flags  "-pedantic")
    set(_loud_flags  "-Wshadow -Wextra")
    set(_quiet_flags "-Wno-unused-function -Wno-unused-variable")
    if(CMAKE_C_COMPILER_IS_GNU)
        add(_def_cxx      "-Wno-unused-but-set-variable -Wno-unused-local-typedefs")
        add(_fast_flags   "-ftree-vectorize -ftree-loop-vectorize")
    else()
        add(_def_cxx "-Qunused-arguments")
        INCLUDE_DIRECTORIES("/usr/include/libcxxabi")
    endif()

    option(ENABLE_GCOV "Enable compilation flags for GNU coverage tool (gcov)" OFF)
    mark_as_advanced(ENABLE_GCOV)
    if(ENABLE_GCOV)
        add(_def_cxx "-fprofile-arcs -ftest-coverage")
    endif(ENABLE_GCOV)

    set(CMAKE_C_FLAGS_INIT                "-W -Wall ${_def_cxx}")
    set(CMAKE_C_FLAGS_DEBUG_INIT          "-g -DDEBUG ${_loud_flags}")
    set(CMAKE_C_FLAGS_MINSIZEREL_INIT     "-Os -DNDEBUG ${_quiet_flags}")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "-g -O2 ${_fast_flags}")
    set(CMAKE_C_FLAGS_RELEASE_INIT        "-O3 -DNDEBUG ${_fast_flags} ${_quiet_flags}")

#------------------------------------------------------------------------------#
# Intel C++ Compilers
#
elseif(CMAKE_C_COMPILER_IS_INTEL)

    set(_std_flags "-Wno-unknown-pragmas -Wno-deprecated")
    set(_extra_flags "-Wno-non-virtual-dtor -Wpointer-arith -Wwrite-strings -fp-model precise")
    set(_par_flags "-parallel-source-info=2")

    get_intel_intrinsic_include_dir()

    set(CMAKE_C_FLAGS_INIT                "${_std_flags} ${_extra_flags} $ENV{C_FLAGS}")
    set(CMAKE_C_FLAGS_DEBUG_INIT          "-debug -DDEBUG ${_par_flags}")
    set(CMAKE_C_FLAGS_MINSIZEREL_INIT     "-Os -DNDEBUG")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "-O2 -debug ${_par_flags}")
    set(CMAKE_C_FLAGS_RELEASE_INIT        "-Ofast -DNDEBUG")

#-----------------------------------------------------------------------
# IBM xlC compiler
#
elseif(CMAKE_C_COMPILER_IS_XLC)

    set(CMAKE_C_FLAGS_INIT "$ENV{C_FLAGS}")
    set(CMAKE_C_FLAGS_DEBUG_INIT          "-g -qdbextra -qcheck=all -qfullpath -qtwolink -+")
    set(CMAKE_C_FLAGS_MINSIZEREL_INIT     "-O2 -qtwolink -+")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "-O2 -g -qdbextra -qcheck=all -qfullpath -qtwolink -+")
    set(CMAKE_C_FLAGS_RELEASE_INIT        "-O2 -qtwolink -+")

#---------------------------------------------------------------------
# HP aC++ Compiler
#
elseif(CMAKE_C_COMPILER_IS_HP_ACC)

    set(CMAKE_C_FLAGS_INIT                "+DAportable +W823 $ENV{C_FLAGS}")
    set(CMAKE_C_FLAGS_DEBUG_INIT          "-g")
    set(CMAKE_C_FLAGS_MINSIZEREL_INIT     "-O3 +Onolimit")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "-O3 +Onolimit -g")
    set(CMAKE_C_FLAGS_RELEASE_INIT        "+O2 +Onolimit")

#---------------------------------------------------------------------
# IRIX MIPSpro CC Compiler
#
elseif(CMAKE_C_COMPILER_IS_MIPS)

    set(CMAKE_C_FLAGS_INIT                "-ptused -DSOCKET_IRIX_SOLARIS")
    set(CMAKE_C_FLAGS_DEBUG_INIT          "-g")
    set(CMAKE_C_FLAGS_MINSIZEREL_INIT     "-O -OPT:Olimit=5000")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "-O -OPT:Olimit=5000 -g")
    set(CMAKE_C_FLAGS_RELEASE_INIT        "-O -OPT:Olimit=5000")

endif()


foreach(_init DEBUG RELEASE MINSIZEREL RELWITHDEBINFO)
    if(NOT "${CMAKE_C_FLAGS_${_init}_INIT}" STREQUAL "")
        string(REPLACE "  " " " CMAKE_C_FLAGS_${_init}_INIT "${CMAKE_C_FLAGS_${_init}_INIT}")
    endif()
endforeach()


