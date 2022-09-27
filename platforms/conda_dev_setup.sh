#!/bin/bash

# If you would like to create your environments in an alternate
# directory, first load the base environment and add the
# alternate location to your conda config:
#
#    conda config --append envs_dirs path/to/envs
#
# Then run this script with the full location:
#
#    ./platforms/conda_dev_setup.sh path/to/envs/my_env
#
# and then when activating the environment you can reference
# it with just "my_env".
#
# NOTE:  by default, the full path to the conda env will be
# displayed in the shell prompt.  To avoid this, edit ~/.condarc
# and add this line:
#
# env_prompt: '({name}) '
#
# Now only the basename of the environment will show up.

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
    echo "Usage:  $0 <name of conda env or path>"
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
pkglist=$(cat "${pkgfile}" | xargs -I % echo -n '"%" ')
platform=$(python -c 'import sys; print(sys.platform)')
if [ "${platform}" = "darwin" ]; then
    pkglist="${pkglist} compilers"
else
    pkglist="${pkglist} gcc_linux-64 gxx_linux-64"
fi

env_noslash=$(echo "${envname}" | sed -e 's/\///g')
is_path=no
if [ "${env_noslash}" != "${envname}" ]; then
    # This was a path
    is_path=yes
    env_check=""
    if [ -e "${envname}/bin/conda" ]; then
	# It already exists
	env_check="${envname}"
    fi
else
    env_check=$(conda env list | grep ${envname})    
fi
    
pip_list="pixell"

if [ "x${env_check}" = "x" ]; then
    # Environment does not yet exist.  Create it.
    if [ ${is_path} = "no" ]; then
	echo "Creating new environment \"${envname}\""
	conda create --yes -n "${envname}"
    else
	echo "Creating new environment \"${envname}\""
	conda create --yes -p "${envname}"
    fi
    echo "Activating environment \"${envname}\""
    conda activate "${envname}"
    echo "Setting default channel in this env to conda-forge"
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict
else
    echo "Activating environment \"${envname}\""
    conda activate "${envname}"
    conda env list
fi

echo "Installing conda packages:  ${pkglist}"
conda install --yes --update-all ${pkglist}
echo "Installing pip packages:  ${piplist}"
pip install ${pip_list}
