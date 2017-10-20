#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/general.sh

export vrun="amplxe-cl \
    -collect concurrency \
    -knob sampling-interval=10 \
    -knob analyze-openmp=true \
    -knob enable-user-tasks=false \
    -knob enable-user-sync=false \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

