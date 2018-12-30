# - Find the name mangling used in the LAPACK library
#
#   Copyright (c) 2019, Ted Kisner
#
# Usage:
#   find_package(LAPACKnames)
#
# It sets the following variables:
#   LAPACK_NAMES     ... equals "UPPER", "LOWER", "UBACK", or "UFRONT".

set(MANGLING_OPTIONS
  "UPPER"
  "LOWER"
  "UBACK"
  "UFRONT")

set(LAPACK_NAMES "")

foreach(MANGLING IN LISTS MANGLING_OPTIONS)
  set(TRY_MANGLING_FLAGS "${LAPACK_LINKER_FLAGS}")
  set(TRY_MANGLING_LIBRARIES "${LAPACK_LIBRARIES}")
  try_compile(TRY_MANGLING ${CMAKE_BINARY_DIR}/tmpLapack
              ${CMAKE_MODULE_PATH}/lapack_mangling.cpp
              CMAKE_FLAGS "${TRY_MANGLING_FLAGS}"
              "-DLINK_LIBRARIES=${TRY_MANGLING_LIBRARIES}"
              COMPILE_DEFINITIONS "-DLAPACK_${MANGLING}")
  if(TRY_MANGLING)
    set(LAPACK_NAMES "${MANGLING}")
    break()
  endif()
endforeach()

if(LAPACK_NAMES STREQUAL "")
    message(SEND_ERROR "Impossible to detect LAPACK name mangling...")
else()
    message(STATUS "LAPACK: detected name mangling: ${LAPACK_NAMES}")

endif()
