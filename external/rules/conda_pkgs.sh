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
    memory_profiler \
    && conda install --copy --yes -c defaults \
    ipython ipython-notebook \
    && python -c "import matplotlib.font_manager" \
    && rm -rf @CONDA_PREFIX@/pkgs/*
