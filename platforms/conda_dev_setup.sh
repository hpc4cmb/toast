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

# Location of build helper tools
condapkgdir=$(dirname "${scriptdir}")/packaging/conda

conda_exe=$(which conda)
if [ "x${conda_exe}" = "x" ]; then
    echo "No conda executable found"
    exit 1
fi

usage () {
    echo ""
    echo "Usage:  $0 <name of conda env or path>"
    echo ""
    echo "The named environment will be activated (and created if needed)"
    echo ""
}

if [ "x${envname}" = "x" ]; then
    usage
    exit 1
fi

eval "${condapkgdir}/install_deps_conda.sh" "${envname}"
