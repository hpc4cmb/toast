#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/general.sh

export vrun="amplxe-cl \
    -collect advanced-hotspots \
    -knob collection-detail=hotspots-sampling \
    -knob event-mode=all \
    -knob analyze-openmp=true \
    -knob sampling-interval=25 \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

