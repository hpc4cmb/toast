#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/default.sh

exort vtune_collect="hotspots"

export vrun="amplxe-cl \
    -collect ${vtune_collect} \
    -knob analyze-openmp=true \
    -knob sampling-interval=10 \
    ${DEFAULT_NATIVE_VTUNE_FLAGS}"

export vtune_collect="$(echo ${vtune_collect} | sed 's/-/_/g')"

