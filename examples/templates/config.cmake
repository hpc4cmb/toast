
MESSAGE(STATUS "Including machine file: \"${MACHFILE}\"...")
INCLUDE("${MACHFILE}")

MESSAGE(STATUS "Including parameter file: \"${SIZEFILE}\"...")
INCLUDE("${SIZEFILE}")

MESSAGE(STATUS "QUEUE: ${QUEUE}")

IF(NOT EXISTS "${INFILE}")
    MESSAGE(FATAL_ERROR "Input file \"${INFILE}\" does not exist")
ENDIF(NOT EXISTS "${INFILE}")

MESSAGE(STATUS "Outputting SLURM file: \"${OUTFILE}\"...")
CONFIGURE_FILE("${INFILE}" "${OUTFILE}" @ONLY)
