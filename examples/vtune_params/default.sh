#!/bin/bash

: ${VTUNE_DATA_LIMIT:=0}
export VTUNE_DATA_LIMIT

export DEFAULT_PYTHON_VTUNE_FLAGS="-mrte-mode=mixed -run-pass-thru=--no-altstack -trace-mpi \
    -finalization-mode=deferred -data-limit=${VTUNE_DATA_LIMIT}"

export DEFAULT_NATIVE_VTUNE_FLAGS="-mrte-mode=native -trace-mpi \
    -finalization-mode=deferred -data-limit=${VTUNE_DATA_LIMIT}"

toupper()
{ 
    echo "$@" | awk -F \|~\| '{print toupper($1)}'
}

tolower()
{ 
    echo "$@" | awk -F \|~\| '{print tolower($1)}'
}

