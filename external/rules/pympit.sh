git clone https://github.com/tskisner/pympit.git \
    && cd pympit \
    && python setup.py build \
    && python setup.py install --prefix=@AUX_PREFIX@ \
    && cd compiled \
    && CC=@MPICC@ make \
    && cp pympit_compiled "@AUX_PREFIX@/bin/" \
    && cd ../.. \
    && rm -rf pympit
