#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/default.sh

export vtune_collect="tsx-exploration"

export vrun="amplxe-cl \
    -collect ${vtune_collect} \
    -knob analysis-step=cyces \
    -knob enable-user-task=false \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

export vtune_collect="$(echo ${vtune_collect} | sed 's/-/_/g')"

