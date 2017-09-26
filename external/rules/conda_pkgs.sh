conda install --copy --yes \
    nose \
    cython \
    numpy \
    scipy \
    matplotlib \
    pyyaml \
    astropy \
    h5py \
    psutil \
    ephem \
    virtualenv \
    pandas \
    memory_profiler \
    ipython \
    && python -c "import matplotlib.font_manager" \
    && rm -rf @CONDA_PREFIX@/pkgs/*
