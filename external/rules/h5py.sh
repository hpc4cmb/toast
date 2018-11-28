curl -SL https://files.pythonhosted.org/packages/74/5d/6f11a5fffc3d8884bb8d6c06abbee0b3d7c8c81bde9819979208ba823a47/h5py-2.8.0.tar.gz \
    | tar xzf - \
    && cd h5py-2.8.0 \
    && CC="@CC@" LDSHARED="@CC@ -shared" \
    python setup.py install --prefix=@AUX_PREFIX@ \
    && cd .. \
    && rm -rf h5py*
