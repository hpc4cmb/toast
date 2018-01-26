# Copyright (c) 2017-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

try:
    import pysm
    available = True
except ModuleNotFoundError:
    pysm = None
    available = False

from .. import timing as timing


class PySMSky(object):
    """
    Create a bandpass-integrated sky map with PySM

    Args:
        PySM input paths / parameters:  FIXME.
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        weights (str): the name of the cache object (<weights>_<detector>)
            containing the pointing weights to use.
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
        units (str): Output units.
    """

    def __init__(self, comm=None, pixels='pixels',
                 out='pysm', nside=None, pysm_sky_config=None, init_sky=True,
                 local_pixels=None, units='K_CMB'):
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self._nside = nside
        self._pixels = pixels
        self._out = out
        self._local_pixels = local_pixels
        self._comm = comm
        self._units = units

        self.pysm_sky_config = pysm_sky_config
        self.sky = self.init_sky(self.pysm_sky_config) if init_sky else None

    def init_sky(self, pysm_sky_config):
        if pysm is None:
            raise RuntimeError('pysm not available')
        autotimer = timing.auto_timer(type(self).__name__)
        initialized_sky_config = {}
        for name, model_id in pysm_sky_config.items():
            initialized_sky_config[name] = \
                pysm.nominal.models(model_id, self._nside, self._local_pixels,
                                    mpi_comm=self._comm)
        return pysm.Sky(initialized_sky_config, mpi_comm=self._comm)

    def exec(self, local_map, out, bandpasses=None):

        if pysm is None:
            raise RuntimeError('pysm not available')

        autotimer = timing.auto_timer(type(self).__name__)
        if self.sky is None:
            self.sky = self.init_sky(self.pysm_sky_config)

        pysm_instrument_config = {
            'beams': None,
            'nside': self._nside,
            'use_bandpass': True,
            'channels': None,
            'channel_names': list(bandpasses.keys()),
            'add_noise': False,
            'output_units': self._units,
            'use_smoothing': False,
            'pixel_indices': self._local_pixels
        }

        for ch_name, bandpass in bandpasses.items():
            pysm_instrument_config["channels"] = [bandpass]
            instrument = pysm.Instrument(pysm_instrument_config)
            out_name = (out + "_" + ch_name) if ch_name else out
            # output of observe is a tuple, first item is the map
            # however the map has 1 extra dimension of size 1,
            # so we need to index [0] twice to get a map of (3, npix)
            local_map[out_name] = \
                instrument.observe(self.sky, write_outputs=False)[0][0]
            assert local_map[out_name].shape[0] == 3
            assert local_map[out_name].ndim == 2
