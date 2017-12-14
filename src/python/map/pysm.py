

class PySMSky(object):
    """
    Create a bandpass integrated sky map with PySM

    Args:
        PySM input paths / parameters:  FIXME.
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        weights (str): the name of the cache object (<weights>_<detector>)
            containing the pointing weights to use.
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
    """
    def __init__(self, pixels='pixels',
                 out='pysm', nside=None, pysm_sky_config=None, init_sky=True, local_pixels=None):
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self._nside = nside
        self._pixels = pixels
        self._out = out
        self._local_pixels = local_pixels

        self.pysm_sky_config = pysm_sky_config
        self.sky = self.init_sky(self.pysm_sky_config) if init_sky else None

    def init_sky(self, pysm_sky_config):
        import pysm
        initialized_sky_config = {}
        for name, model_id in pysm_sky_config.items():
            initialized_sky_config[name] = \
                pysm.nominal.models(model_id, self._nside, self._local_pixels)
        return pysm.Sky(initialized_sky_config)

    def exec(self, local_map, out, bandpasses=None):

        import pysm

        if self.sky is None:
            self.sky = self.init_sky(self.pysm_sky_config)

        pysm_instrument_config = {
            'beams': None,
            'nside': self._nside,
            'use_bandpass': True,
            'channels': None,
            'channel_names': list(bandpasses.keys()),
            'add_noise': False,
            'output_units': 'uK_RJ',
            'use_smoothing': False,
            'pixel_indices': self._local_pixels
        }

        for ch_name, bandpass in bandpasses.items():
            pysm_instrument_config["channels"] = [bandpass]
            instrument = pysm.Instrument(pysm_instrument_config)
            out_name = (out + "_" + ch_name) if ch_name else out
            local_map[out_name], _ = \
                instrument.observe(self.sky, write_outputs=False)
