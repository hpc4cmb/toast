conda install --copy --yes \
    nose \
    cython \
    numpy \
    scipy \
    matplotlib \
    pyyaml \
    astropy \
    psutil \
    ephem \
    virtualenv \
    pandas \
    memory_profiler \
    ipython \
    cmake \
    && python -c "import matplotlib.font_manager" \
    && rm -rf @CONDA_PREFIX@/pkgs/*
