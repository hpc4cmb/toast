# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

from ..utils import set_numba_threading

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

    Requires PySM 3. It initializes the `pysm.Sky` object either
    in the constructor (`init_sky=True`) or when the `exec` method
    is executed (`init_sky=False`).
    Inizialization of the sky will load templates from the first process
    of the MPI communicator, copy to all processes and then select the local
    rings, distributed as required by `libsharp` for eventual smoothing later.
    If another pixel distribution is required, it can be specified with `pixel_indices`,
    however a different pixel distribution can only be used if performing no smoothing.

    Args:
        comm (toast.Comm): Toast communicator.
        mpi_comm_name (str): The name of the MPI sub communicator to use.
            Valid names are "group" or "rank".
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
        mpi_comm_name=None,
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
        self._units = u.Unit(units)
        self.map_dist = map_dist

        if comm is None:
            raise RuntimeError("You must specify the toast Comm instance.")

        # This is the toast.Comm object
        self._comm = comm

        # This is the name of the MPI sub-communicator that we will use
        # to collectively simulate data
        self._mpi_comm_name = mpi_comm_name

        # Select the MPI communicator based on the name.
        self.mpi_comm = None
        if self._mpi_comm_name == "rank":
            self.mpi_comm = self._comm.comm_rank
        else:
            self.mpi_comm = self._comm.comm_group

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
                if self.mpi_comm is None
                else pysm.MapDistribution(
                    pixel_indices=self._pixel_indices,
                    nside=self._nside,
                    mpi_comm=self.mpi_comm,
                )
            )
        # The world communicator
        world_comm = self._comm.comm_world

        if (world_comm is None) or (world_comm.rank == 0):
            # Instantiating a Sky object may trigger downloading and
            # caching of data with astropy.  We do that once here on a single
            # process to avoid concurrent downloading of the same file.
            temp_sky = pysm.Sky(
                nside=self._nside,
                preset_strings=pysm_sky_config,
                component_objects=self.pysm_component_objects,
                map_dist=None,
                output_unit=self._units,
            )
            del temp_sky
        if world_comm is not None:
            world_comm.barrier()

        # Now load and return a Sky object.  Even though we have cached the
        # data with astropy, concurrent reads of the data seem to produce
        # astropy locking errors.  So here we ensure that only one process
        # at a time reads the cached data.  If our Sky object is using the
        # "group" communicator, then serialize the construction across groups.
        # If we are using the "rank" communicator, then serialize across the
        # ranks.
        pysm_sky = None
        reader_comm = None
        if self._mpi_comm_name == "rank":
            reader_comm = self._comm.comm_group
        else:
            reader_comm = self._comm.comm_rank
        if reader_comm is None:
            # Only one reader, no need to take turns
            pysm_sky = pysm.Sky(
                nside=self._nside,
                preset_strings=pysm_sky_config,
                component_objects=self.pysm_component_objects,
                map_dist=self.map_dist,
                output_unit=self._units,
            )
        else:
            for reader in range(reader_comm.size):
                if reader == reader_comm.rank:
                    # My turn
                    pysm_sky = pysm.Sky(
                        nside=self._nside,
                        preset_strings=pysm_sky_config,
                        component_objects=self.pysm_component_objects,
                        map_dist=self.map_dist,
                        output_unit=self._units,
                    )
                reader_comm.barrier()
        return pysm_sky

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
