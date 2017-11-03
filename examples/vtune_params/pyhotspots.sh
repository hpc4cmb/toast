#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/default.sh

export vtune_collect="pyhotspots"

export vrun="amplxe-cl \
    -collect hotspots \
    -knob sampling-interval=25 \
    -knob analyze-openmp=true \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

export vtune_collect="$(echo ${vtune_collect} | sed 's/-/_/g')"

