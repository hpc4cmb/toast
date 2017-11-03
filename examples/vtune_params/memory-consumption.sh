#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/default.sh

export vtune_collect="memory-consumption"

export vrun="amplxe-cl \
    -collect ${vtune_collect} \
    -knob mem-object-size-min-thres=1024 \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

export vtune_collect="$(echo ${vtune_collect} | sed 's/-/_/g')"

