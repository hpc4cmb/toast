#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/general.sh

export vrun="amplxe-cl \
    -collect general-exploration \
    -knob collect-memory-bandwidth=true \
    -knob enable-user-tasks=false \
    -knob dram-bandwidth-limit=true \
    -knob analyze-openmp=false \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

