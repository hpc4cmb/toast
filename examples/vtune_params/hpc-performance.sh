#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/default.sh

vtune_collect="hpc-performance"

export vrun="amplxe-cl \
    -collect ${vtune_collect} \
    -knob sampling-interval=25 \
    -knob enable-stack-collection=false \
    -knob collect-memory-bandwidth=false \
    -knob dram-bandwidth-limits=false \
    -knob analyze-openmp=true \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

export vtune_collect="$(echo ${vtune_collect} | sed 's/-/_/g')"

