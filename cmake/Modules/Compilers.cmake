
################################################################################
#
#        Compilers
#
################################################################################
#
#   sets (cached):
#
#       CMAKE_C_COMPILER_IS_<TYPE>
#       CMAKE_CXX_COMPILER_IS_<TYPE>
#
#
#
#

#   include guard
if(__compilers_is_loaded)
    return()
endif()
set(__compilers_is_loaded ON)

#   languages
foreach(LANG C CXX)

    macro(SET_COMPILER_VAR VAR _BOOL)
        set(CMAKE_${LANG}_COMPILER_IS_${VAR} ${_BOOL} CACHE STRING
            "CMake ${LANG} compiler identification (${VAR})")
        mark_as_advanced(CMAKE_${LANG}_COMPILER_IS_${VAR})
    endmacro()

    if(("${LANG}" STREQUAL "C" AND CMAKE_COMPILER_IS_GNUCC)
        OR
       ("${LANG}" STREQUAL "CXX" AND CMAKE_COMPILER_IS_GNUCXX))

        # GNU compiler
        SET_COMPILER_VAR(       GNU                 ON)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "icc.*")

        # Intel icc compiler
        SET_COMPILER_VAR(       INTEL               ON)
        SET_COMPILER_VAR(       INTEL_ICC           ON)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "icpc.*")

        # Intel icpc compiler
        SET_COMPILER_VAR(       INTEL               ON)
        SET_COMPILER_VAR(       INTEL_ICPC          ON)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "Clang")

        # Clang/LLVM compiler
        SET_COMPILER_VAR(       CLANG               ON)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "PGI")

        # PGI compiler
        SET_COMPILER_VAR(       PGI                 ON)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "xlC" AND UNIX)

        # IBM xlC compiler
        SET_COMPILER_VAR(       XLC                 ON)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "aCC" AND UNIX)

        # HP aC++ compiler
        SET_COMPILER_VAR(       HP_ACC              ON)

    elseif(CMAKE_${LANG}_COMPILER MATCHES "CC" AND CMAKE_SYSTEM_NAME MATCHES "IRIX" AND UNIX)

        # IRIX MIPSpro CC Compiler
        SET_COMPILER_VAR(       MIPS                ON)

    elseif(CMAKE_${LANG}_COMPILER_ID MATCHES "Intel")

        SET_COMPILER_VAR(       INTEL               ON)

        set(CTYPE ICC)
        if("${LANG}" STREQUAL "CXX")
            set(CTYPE ICPC)
        endif("${LANG}" STREQUAL "CXX")

        SET_COMPILER_VAR(       INTEL_${CTYPE}      ON)

    endif()

    # set other to no
    foreach(TYPE GNU INTEL INTEL_ICC INTEL_ICPC CLANG PGI XLC HP_ACC MIPS)
        if(${CMAKE_${LANG}_COMPILER_IS_${TYPE}})
            continue()
        else()
            SET_COMPILER_VAR(${TYPE} OFF)
        endif()
    endforeach()
endforeach()

