#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/default.sh

export vtune_collect="advanced-hotspots"

export vrun="amplxe-cl \
    -collect ${vtune_collect} \
    -knob collection-detail=hotspots-sampling \
    -knob event-mode=all \
    -knob analyze-openmp=true \
    -knob sampling-interval=25 \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

export vtune_collect="$(echo ${vtune_collect} | sed 's/-/_/g')"

