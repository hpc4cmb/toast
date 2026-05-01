#!/bin/bash

# Helper script to create a conda environment with all of the
# toast build dependencies.  See installation section of the
# documentation.  This script is only needed if you are building
# toast from source.
#
# If you would like to create your environments in an alternate
# directory, first load the base environment and add the
# alternate location to your conda config:
#
#    conda config --append envs_dirs path/to/envs
#
# Then run this script with the full location:
#
#    ./conda_setup.sh path/to/envs/my_env
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

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1

show_help () {
    echo "" >&2
    echo "Usage:  $0" >&2
    echo "    [-e <environment, either name or full path>]" >&2
    echo "    [-p <python version (e.g. 3.12)>]" >&2
    echo "    [-x (install extra dependencies)]" >&2
    echo "" >&2
    echo "    Create a conda environment for TOAST development." >&2
    echo "" >&2
    echo "" >&2
}

envname=""
pyversion=$(python3 --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
extra="no"

while getopts ":e:p:x" opt; do
    case $opt in
        e)
            envname="${OPTARG}"
            ;;
        p)
            pyversion="${OPTARG}"
            ;;
        x)
            extra="yes"
            ;;
        \?)
            show_help
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            show_help
            exit 1
            ;;
    esac
done

shift $((OPTIND-1))

if [ -z "${envname}" ]; then
    echo ""
    echo "You must specify the environment name or path!"
    show_help
    exit 1
fi

# Location of build helper tools
condapkgdir="${scriptdir}/packaging/conda"

conda_exe=$(which conda)
if [ "x${conda_exe}" = "x" ]; then
    echo "No conda executable found"
    exit 1
fi

eval "${condapkgdir}/install_deps_conda.sh" "${envname}" "${pyversion}" "${extra}"
