#!/bin/bash
#
# Run this script to launch the travis docker container.  Go to a recent
# travis build of toast and find the "instance" used for that build (i.e.
# the docker image).  It is located under the "worker" section and should
# look like "travisci/ci-garnet:packer-1512502276-986baf0".
#
# Pass this string as the first argument to this script.  Run this script from
# this directory!
#
#    %> ./travis_docker.sh travisci/ci-garnet:packer-1512502276-986baf0
#

usage () {
    echo "$0 <travis docker instance>"
    exit 1
}

inst="$1"
if [ "x${inst}" = "x" ]; then
    usage
    exit 1
fi

build_id="build-${RANDOM}"
scrsource=$(pwd)
scrtarget="/home/travis/scripts"

docker run --name ${build_id} --mount type=bind,source=${scrsource},target=${scrtarget} -dit ${inst} /sbin/init

docker exec -it ${build_id} bash -l -c 'su - travis'
