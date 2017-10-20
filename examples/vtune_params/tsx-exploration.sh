#!/bin/bash

TWD="$(dirname ${BASH_SOURCE[0]})"

. ${TWD}/finalize.sh
. ${TWD}/general.sh

export vrun="amplxe-cl \
    -collect tsx-exploration \
    -knob analysis-step=cyces \
    -knob enable-user-task=false \
    ${DEFAULT_PYTHON_VTUNE_FLAGS}"

