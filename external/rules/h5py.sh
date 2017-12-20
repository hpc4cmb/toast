curl -SL https://pypi.python.org/packages/41/7a/6048de44c62fc5e618178ef9888850c3773a9e4be249e5e673ebce0402ff/h5py-2.7.1.tar.gz#md5=da630aebe3ab9fa218ac405a218e95e0 \
    | tar xzf - \
    && cd h5py-2.7.1 \
    && CC="@CC@" LDSHARED="@CC@ -shared" \
    python setup.py install --prefix=@AUX_PREFIX@ \
    && cd .. \
    && rm -rf h5py*
