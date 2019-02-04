# - FindMPI4PY
# Find mpi4py includes and library
# This module defines:
# MPI4PY_INCLUDE_DIR, where to find mpi4py.h, etc.
# MPI4PY_FOUND

# Roughly based on a file in a branch of the Dolfin project.

if(NOT MPI4PY_INCLUDE_DIR)
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/cmake/test_mpi4py.py"
        OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
        RESULT_VARIABLE MPI4PY_COMMAND_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(MPI4PY_COMMAND_RESULT STREQUAL "0")
        message("-- Found mpi4py include: ${MPI4PY_INCLUDE_DIR}")
        set(MPI4PY_FOUND TRUE)
        set(MPI4PY_INCLUDE_DIR ${MPI4PY_INCLUDE_DIR}
            CACHE STRING "mpi4py include path")
    else(MPI4PY_COMMAND_RESULT STREQUAL "0")
        message("-- mpi4py NOT found")
        set(MPI4PY_FOUND FALSE)
    endif(MPI4PY_COMMAND_RESULT STREQUAL "0")
else(NOT MPI4PY_INCLUDE_DIR)
    message("-- Using mpi4py include: ${MPI4PY_INCLUDE_DIR}")
    set(MPI4PY_FOUND TRUE)
endif(NOT MPI4PY_INCLUDE_DIR)
