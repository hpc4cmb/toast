# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..timing import function_timer

pysm = None
try:
    import pysm
    import pysm.units as u
except ImportError:
    pysm = None
    u = None


class PySMSky(object):
    """Create a bandpass-integrated sky map with PySM

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

    def __init__(
        self,
        comm=None,
        pixels="pixels",
        out="pysm",
        nside=None,
        pysm_sky_config=None,
        pysm_precomputed_cmb_K_CMB=None,
        init_sky=True,
        pixel_indices=None,
        units="K_CMB",
    ):
        self._nside = nside
        self._pixels = pixels
        self._out = out
        self._pixel_indices = pixel_indices
        self._comm = comm
        self._units = u.Unit(units)

        self.pysm_sky_config = pysm_sky_config
        self.pysm_precomputed_cmb_K_CMB = pysm_precomputed_cmb_K_CMB
        self.sky = (
            self.init_sky(self.pysm_sky_config, self.pysm_precomputed_cmb_K_CMB)
            if init_sky
            else None
        )

    @function_timer
    def init_sky(self, pysm_sky_config, pysm_precomputed_cmb_K_CMB):
        if pysm is None:
            raise RuntimeError("pysm not available")
        if pysm_precomputed_cmb_K_CMB is not None:
            pass
            # cmb = {
            #     "model": "pre_computed",
            #     "nside": self._nside,
            #     "pixel_indices": self._pixel_indices,
            # }
            # # PySM expects uK_CMB
            # cmb["A_I"], cmb["A_Q"], cmb["A_U"] = (
            #     np.array(
            #         pysm.read_map(
            #             pysm_precomputed_cmb_K_CMB,
            #             self._nside,
            #             field=(0, 1, 2),
            #             pixel_indices=self._pixel_indices,
            #             mpi_comm=self._comm,
            #         )
            #     )
            #     * 1e6
            # )
            # initialized_sky_config["cmb"] = [cmb]
            # # remove cmb from the pysm string
            # pysm_sky_config.pop("cmb", None)
        map_dist = pysm.MapDistribution(
            pixel_indices=self._pixel_indices, nside=self._nside, mpi_comm=self._comm
        )
        return pysm.Sky(
            nside=self._nside,
            preset_strings=list(pysm_sky_config.values()),
            map_dist=map_dist,
            output_unit=self._units,
        )

    @function_timer
    def exec(self, local_map, out, bandpasses=None):

        if pysm is None:
            raise RuntimeError("pysm not available")

        if self.sky is None:
            self.sky = self.init_sky(
                self.pysm_sky_config, self.pysm_precomputed_cmb_K_CMB
            )

        for ch_name, bandpass in bandpasses.items():
            out_name = (out + "_" + ch_name) if ch_name else out
            local_map[out_name] = self.sky.get_emission(
                bandpass[0] * u.GHz, bandpass[1]
            ).value
            assert local_map[out_name].shape[0] == 3
            assert local_map[out_name].ndim == 2
