#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/general.sh

export vrun="amplxe-cl \
    -collect tsx-hotspots \
    -knob sampling-interval=10 \
    -knob enable-stack-collection=true \
    -finalization-mode=deferred \
    -trace-mpi \
    -data-limit=${VTUNE_DATA_LIMIT}"

