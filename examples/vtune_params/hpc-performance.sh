#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/general.sh

export vrun="amplxe-cl \
    -collect hpc-performance \
    -knob sampling-interval=50 \
    -knob enable-stack-collection=false \
    -knob collect-memory-bandwidth=false \
    -knob dram-bandwidth-limits=false \
    -knob analyze-openmp=true \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

