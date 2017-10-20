#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/general.sh

export vrun="amplxe-cl \
    -collect locksandwaits \
    -knob sampling-interval=100 \
    -knob enable-user-tasks=false \
    -knob enable-user-sync=false \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

