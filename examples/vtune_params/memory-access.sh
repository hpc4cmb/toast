#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/general.sh

export vrun="amplxe-cl \
    -collect memory-access \
    -knob sampling-interval=100 \
    -knob analyze-mem-objects=true \
    -knob mem-object-size-min-thres=1024 \
    -knob dram-bandwidth-limits=true \
    -knob analyze-openmp=false \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

