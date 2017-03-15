conda install --copy --yes \
    nose \
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
    && python -c "import matplotlib.font_manager" \
    && rm -rf @CONDA_PREFIX@/pkgs/*
