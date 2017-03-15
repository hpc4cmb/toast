curl -SL https://pypi.python.org/packages/ee/b8/f443e1de0b6495479fc73c5863b7b5272a4ece5122e3589db6cd3bb57eeb/mpi4py-2.0.0.tar.gz#md5=4f7d8126d7367c239fd67615680990e3 \
    -o mpi4py-2.0.0.tar.gz \
    && tar xzf mpi4py-2.0.0.tar.gz \
    && cd mpi4py-2.0.0 \
    && python setup.py build --mpicc="@MPICC@" --mpicxx="@MPICXX@" \
    && python setup.py install --prefix=@AUX_PREFIX@ \
    && cd .. \
    && rm -rf mpi4py*
