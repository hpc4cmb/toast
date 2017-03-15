conda config --add channels intel \
    && conda install --copy --yes intelpython3_core \
    && rm -rf @CONDA_PREFIX@/pkgs/*
