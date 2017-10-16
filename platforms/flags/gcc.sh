#!/bin/bash

WARN_FLAGS="-W -Wall -Wwrite-strings -Wpointer-arith"
WARN_FLAGS="${WARN_FLAGS} -pedantic -Wshadow -Wextra"
NOWARN_FLAGS="-Wno-unused-parameter -Wno-deprecated -Wno-unused-variable"
NOWARN_FLAGS="${NOWARN_FLAGS} -Wno-unused-function -Wno-expansion-to-defined"
NOWARN_FLAGS="${NOWARN_FLAGS} -Wno-vla -Wno-unused-but-set-variable"
C_WARN_FLAGS="${WARN_FLAGS} ${NOWARN_FLAGS}"
CXX_WARN_FLAGS="${WARN_FLAGS} -Woverloaded-virtual ${NOWARN_FLAGS}"

export C_WARN_FLAGS
export CXX_WARN_FLAGS
