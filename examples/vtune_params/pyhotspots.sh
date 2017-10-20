#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/general.sh

export vrun="amplxe-cl \
    -collect hotspots \
    -knob sampling-interval=500 \
    -run-pass-thru=--no-altstack \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

