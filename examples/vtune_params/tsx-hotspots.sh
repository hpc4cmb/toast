#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/default.sh

export vtune_collect="tsx-hotspots"

export vrun="amplxe-cl \
    -collect ${vtune_collect} \
    -knob sampling-interval=10 \
    -knob enable-stack-collection=true \
    -finalization-mode=deferred \
    -trace-mpi \
    -data-limit=${VTUNE_DATA_LIMIT}"

export vtune_collect="$(echo ${vtune_collect} | sed 's/-/_/g')"

