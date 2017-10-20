#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/general.sh

export vrun="amplxe-cl \
    -collect hotspots \
    -knob analyze-openmp=true \
    -knob sampling-interval=10 \
    ${DEFAULT_NATIVE_VTUNE_FLAGS}"

