#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/default.sh

export vtune_collect="locksandwaits"

export vrun="amplxe-cl \
    -collect ${vtune_collect} \
    -knob sampling-interval=25 \
    -knob enable-user-tasks=false \
    -knob enable-user-sync=false \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

export vtune_collect="$(echo ${vtune_collect} | sed 's/-/_/g')"

