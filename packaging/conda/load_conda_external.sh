# This shell function loads a conda environment as well as
# the associated externally compiled packages.

prepend_ext_env () {
    # This function is needed since trailing colons
    # on some environment variables can cause major
    # problems...
    local envname="$1"
    local envval="$2"
    eval "local temp=\"\${$envname}\""
    if [ -z ${temp+x} ]; then
        export ${envname}="${envval}"
    else
        export ${envname}="${envval}:${temp}"
    fi
}

load_conda_ext () {
    envname=$1
    conda activate "${envname}"
    extprefix="${CONDA_PREFIX}_ext"
    prepend_ext_env "PATH" "${extprefix}/bin"
    prepend_ext_env "CPATH" "${extprefix}/include"
    prepend_ext_env "LIBRARY_PATH" "${extprefix}/lib"
    prepend_ext_env "LD_LIBRARY_PATH" "${extprefix}/lib"
}
