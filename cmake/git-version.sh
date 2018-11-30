#!/bin/bash

fullstr=$(git describe --tags --dirty --always)
tagstr=$(git describe --tags --dirty --always | cut -d "-" -f 1)
cnt=$(git rev-list --count HEAD)

if [ "${fullstr}" = "${tagstr}" ]; then
    # we are at a tag
    echo "${tagstr}"
else
    # build compatible version
    echo "${tagstr}.dev${cnt}"
fi
