# Variables set by calling code:
# - CONDA_EXE
# - ENVNAME
#

# Activate the base environment
basedir=$(dirname $(dirname "${CONDA_EXE}"))
source ${basedir}/etc/profile.d/conda.sh
conda deactivate
conda activate base

conda env list

# Determine whether the environment is a name or a
# full path.
env_noslash=$(echo "${ENVNAME}" | sed -e 's/\///g')
echo "env_noslash = ${env_noslash}"

is_path=no
if [ "${env_noslash}" != "${ENVNAME}" ]; then
    # This was a path
    echo "is a path"
    is_path=yes
    env_check=""
    if [ -e "${ENVNAME}/bin/conda" ]; then
        # It already exists
        echo "path exists"
        env_check="${ENVNAME}"
    fi
else
    echo "checking list for ${ENVNAME}"
    env_check=$(conda env list | { grep "${ENVNAME} " || true; })
fi

echo "env_check = ${env_check}"

if [ "x${env_check}" = "x" ]; then
    # Environment does not yet exist.  Create it.
    echo "Creating new environment \"${ENVNAME}\""
    if [ ${is_path} = "no" ]; then
        conda create --yes -n "${ENVNAME}"
    else
        conda create --yes -p "${ENVNAME}"
    fi
    echo "Activating environment \"${ENVNAME}\""
    conda activate "${ENVNAME}"
    echo "Setting default channel in this env to conda-forge"
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict
else
    echo "Activating environment \"${ENVNAME}\""
    conda activate "${ENVNAME}"
    conda env list
fi

