
IF(NOT "${MACHFILE}" STREQUAL "")
    MESSAGE(STATUS "Including machine file: \"${MACHFILE}\"...")
    INCLUDE("${MACHFILE}")
ENDIF(NOT "${MACHFILE}" STREQUAL "")

MESSAGE(STATUS "Including parameter file: \"${SIZEFILE}\"...")
INCLUDE("${SIZEFILE}")

IF(NOT EXISTS "${INFILE}")
    MESSAGE(FATAL_ERROR "Input file \"${INFILE}\" does not exist")
ENDIF(NOT EXISTS "${INFILE}")

IF(NOT "${READFILE}" STREQUAL "")
    MESSAGE(STATUS "Including read file: \"${READFILE}\"...")
    INCLUDE("${READFILE}")
ENDIF(NOT "${READFILE}" STREQUAL "")

MESSAGE(STATUS "Outputting SLURM file: \"${OUTFILE}\"...")
CONFIGURE_FILE("${INFILE}" "${OUTFILE}" @ONLY)
