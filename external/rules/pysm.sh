git clone https://github.com/zonca/PySM_public.git --branch megarun_2017 --single-branch --depth 1 \
    && cd PySM_public \
    && python setup.py install --prefix=@AUX_PREFIX@ \
    && cd .. \
    && rm -rf PySM*
