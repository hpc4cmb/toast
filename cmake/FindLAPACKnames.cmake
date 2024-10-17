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
  "LOWER"
  "UPPER"
  "UBACK"
  "UFRONT"
)

set(LAPACK_NAMES "")

set(OMP_BLAS "")
if(OpenMP_C_FOUND)
  set(OMP_BLAS "${OpenMP_C_FLAGS}")
endif()

foreach(MANGLING IN LISTS MANGLING_OPTIONS)
  try_compile(TRY_MANGLING ${CMAKE_BINARY_DIR}/tmpLapack
              ${CMAKE_MODULE_PATH}/lapack_mangling.cpp
              CMAKE_FLAGS "${BLAS_LINKER_FLAGS}" "${LAPACK_LINKER_FLAGS}"
              COMPILE_DEFINITIONS "-DLAPACK_${MANGLING}"
              LINK_LIBRARIES "${LAPACK_LIBRARIES}" "${BLAS_LIBRARIES}" "${OMP_BLAS}"
              OUTPUT_VARIABLE OUTPUT_MANGLING)
  message("Test output for LAPACK_${MANGLING}:")
  message(${OUTPUT_MANGLING})
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
