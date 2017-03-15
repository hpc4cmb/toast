conda install --copy --yes \
    cython \
    numpy \
    scipy \
    matplotlib \
    pyyaml \
    astropy \
    h5py \
    jupyter \
    psutil \
    ephem \
    && rm -rf @CONDA_PREFIX@/pkgs/*
