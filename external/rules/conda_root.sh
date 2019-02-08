curl -SL @MINICONDA@ \
    -o miniconda.sh \
    && /bin/bash miniconda.sh -b -f -p @CONDA_PREFIX@ \
    && conda config --add channels intel \
    && conda config --remove channels intel \
    && conda install --copy --yes python=@PYVERSION@ \
    && rm miniconda.sh \
    && rm -rf @CONDA_PREFIX@/pkgs/* \
    && ln -s @CONDA_PREFIX@/lib/libpython* @AUX_PREFIX@/lib/
