curl -SL @MINICONDA@ \
    -o miniconda.sh \
    && /bin/bash miniconda.sh -b -f -p @CONDA_PREFIX@ \
    && conda install --copy --yes python=3.5.2 \
    && rm miniconda.sh \
    && rm -rf @CONDA_PREFIX@/pkgs/*
