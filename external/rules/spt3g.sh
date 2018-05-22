git clone https://github.com/CMB-S4/spt3g_software.git --branch master --single-branch --depth 1 \
    && export spt3g_start=$(pwd) \
    && cd spt3g_software \
    && patch -p1 < ../rules/patch_spt3g \
    && cd .. \
    && cp -a spt3g_software "@AUX_PREFIX@/spt3g" \
    && cd "@AUX_PREFIX@/spt3g" \
    && mkdir build \
    && cd build \
    && LDFLAGS="-Wl,-z,muldefs" \
    cmake \
    -DCMAKE_C_COMPILER="@CC@" \
    -DCMAKE_CXX_COMPILER="@CXX@" \
    -DCMAKE_C_FLAGS="@CFLAGS@" \
    -DCMAKE_CXX_FLAGS="@CXXFLAGS@" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DPYTHON_EXECUTABLE:FILEPATH="@CONDA_PREFIX@/bin/python" \
    .. \
    && make -j @MAKEJ@ \
    && ln -s @AUX_PREFIX@/spt3g/build/bin/* @AUX_PREFIX@/bin/ \
    && ln -s @AUX_PREFIX@/spt3g/build/spt3g @AUX_PREFIX@/lib/python@PYVERSION@/site-packages/ \
    && cd ${spt3g_start} \
    && rm -rf spt3g_software
