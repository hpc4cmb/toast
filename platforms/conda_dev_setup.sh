#!/bin/bash

envname=$1

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1

conda_exe=$(which conda)
if [ "x${conda_exe}" = "x" ]; then
    echo "No conda executable found"
    exit 1
fi

usage () {
    echo ""
    echo "Usage:  $0 <name of conda env>"
    echo ""
    echo "The named environment will be activated (and created)"
    echo ""
}

if [ "x${envname}" = "x" ]; then
    usage
    exit 1
fi

basedir=$(dirname $(dirname "${conda_exe}"))
source ${basedir}/etc/profile.d/conda.sh

conda activate base

pkgfile="${scriptdir}/conda_dev_pkgs.txt"
pkglist=$(cat "${pkgfile}" | xargs echo)
platform=$(python -c 'import sys; print(sys.platform)')
if [ "${platform}" = "darwin" ]; then
    pkglist="${pkglist} compilers"
else
    pkglist="${pkglist} gcc_linux-64 gxx_linux-64"
fi

env_check=$(conda env list | grep ${envname})

pip_list="pixell"

if [ "x${env_check}" = "x" ]; then
    # Environment does not yet exist.  Create it.
    echo "Creating new environment \"${envname}\" with packages:  ${pkglist}"
    conda create --yes -n ${envname} ${pkglist}
    echo "Activating environment \"${envname}\""
    conda activate ${envname}
    echo "Installing pip packages:  ${piplist}"
    pip install ${pip_list}
else
    echo "Activating environment \"${envname}\""
    conda activate ${envname}
    conda env list
    echo "Installing conda packages:  ${pkglist}"
    conda install --yes --update-all ${pkglist}
    echo "Installing pip packages:  ${piplist}"
    pip install ${pip_list}
fi
    

