#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/default.sh

export vtune_collect="memory-access"

export vrun="amplxe-cl \
    -collect ${vtune_collect} \
    -knob sampling-interval=25 \
    -knob analyze-mem-objects=true \
    -knob mem-object-size-min-thres=1024 \
    -knob dram-bandwidth-limits=true \
    -knob analyze-openmp=false \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"


