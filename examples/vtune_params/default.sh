#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh

: ${VTUNE_DATA_LIMIT:=0}
export VTUNE_DATA_LIMIT

export DEFAULT_PYTHON_VTUNE_FLAGS="-mrte-mode=mixed -run-pass-thru=--no-altstack -trace-mpi \
    -finalization-mode=deferred -data-limit=${VTUNE_DATA_LIMIT} ${_VTUNE_SEARCH_DIRS}"

export DEFAULT_NATIVE_VTUNE_FLAGS="-mrte-mode=native -run-pass-thru=--no-altstack -trace-mpi \
    -finalization-mode=deferred -data-limit=${VTUNE_DATA_LIMIT} ${_VTUNE_SEARCH_DIRS}"

toupper()
{ 
    echo "$@" | awk -F \|~\| '{print toupper($1)}'
}

tolower()
{ 
    echo "$@" | awk -F \|~\| '{print tolower($1)}'
}

