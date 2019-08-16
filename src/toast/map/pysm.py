# Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..utils import set_numba_threading

from ..timing import function_timer

# Force the numba threading layer to be OpenMP if possible before importing PySM
set_numba_threading()

pysm = None
try:
    import pysm
    import pysm.units as u
except ImportError:
    pysm = None
    u = None


class PySMSky(object):
    """Create a bandpass-integrated sky map with PySM

    Requires PySM 3. It initializes the `pysm.Sky` object either
    in the constructor (`init_sky=True`) or when the `exec` method
    is executed (`init_sky=False`).
    Inizialization of the sky will load templates from the first process
    of the MPI communicator, copy to all processes and then select the local
    rings, distributed as required by `libsharp` for eventual smoothing later.
    If another pixel distribution is required, it can be specified with `pixel_indices`,
    however a different pixel distribution can only be used if performing no smoothing.

    Args:
        comm (mpi4py.MPI.Comm): MPI communicator
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
        nside (int): :math:`N_{side}` for PySM
        pysm_sky_config (list(str)): list of PySM components, e.g. ["s1", "d1"],
            this will be passed as `preset_strings` to `pysm.Sky`
        pysm_precomputed_cmb_K_CMB (str): obsolete, will be removed shortly
        pysm_component_objects (list(pysm.Model)): extra sky components that
            inherits from `pysm.Model` to be passed to `pysm.Sky` as `components_objects`
        init_sky (bool): Initializes the sky in the constructor if True, in `exec` if False
        pixel_indices (np.array(int)): List of pixel indices, use None to use the standard
            ring-based libsharp distribution
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
        pysm_component_objects=None,
        init_sky=True,
        pixel_indices=None,
        units="K_CMB",
        map_dist=None,
    ):
        self._nside = nside
        self._pixels = pixels
        self._out = out
        self._pixel_indices = pixel_indices
        self._comm = comm
        self._units = u.Unit(units)
        self.map_dist = map_dist

        self.pysm_sky_config = pysm_sky_config
        self.pysm_precomputed_cmb_K_CMB = pysm_precomputed_cmb_K_CMB
        self.pysm_component_objects = pysm_component_objects
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
            raise NotImplementedError(
                "pysm_precomputed_cmb_K_CMB is not currently supported"
            )
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
        if self.map_dist is None:
            self.map_dist = (
                None
                if self._comm is None
                else pysm.MapDistribution(
                    pixel_indices=self._pixel_indices,
                    nside=self._nside,
                    mpi_comm=self._comm,
                )
            )
        return pysm.Sky(
            nside=self._nside,
            preset_strings=pysm_sky_config,
            component_objects=self.pysm_component_objects,
            map_dist=self.map_dist,
            output_unit=self._units,
        )

    @function_timer
    def exec(self, local_map, out, bandpasses=None):
        """Execute PySM

        Executes PySM on the given bandpasses and return the
        bandpass-integrated output maps

        Args:
        local_map (dict): Dictionary that will contain output maps
        out (str): Output maps in `local_map` will be named `out_{chname}`
            where `chname` is the key in the bandpasses dictionary
        bandpasses (dict): Dictionary with channel names as keys and
            a tuple of (frequency, weight) as values, PySM will normalize
            the bandpasses and integrate the signal in Jy/sr

        """

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
