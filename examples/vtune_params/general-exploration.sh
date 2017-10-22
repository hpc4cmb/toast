#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/default.sh

export vtune_collect="general-exploration"

export vrun="amplxe-cl \
    -collect ${vtune_collect} \
    -knob collect-memory-bandwidth=true \
    -knob enable-user-tasks=false \
    -knob dram-bandwidth-limit=true \
    -knob analyze-openmp=false \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

export vtune_collect="$(echo ${vtune_collect} | sed 's/-/_/g')"

