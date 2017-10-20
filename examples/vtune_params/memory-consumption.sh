#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/general.sh

export vrun="amplxe-cl \
    -collect memory-consumption \
    -knob mem-object-size-min-thres=1024 \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

