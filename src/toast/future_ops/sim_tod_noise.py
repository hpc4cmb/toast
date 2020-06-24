# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ..timing import function_timer

from ..fft import FFTPlanReal1DStore

from ..tod.tod_math import sim_noise_timestream

from ..operator import Operator

from ..config import ObjectConfig

from ..utils import rate_from_times, Logger


class SimNoise(Operator):
    """Operator which generates noise timestreams.

    This passes through each observation and every process generates data
    for its assigned samples.  The observation unique ID is used in the random
    number generation.  The observation dictionary can optionally include a
    'global_offset' member that might be useful if you are splitting observations and
    want to enforce reproducibility of a given sample, even when using
    different-sized observations.

    Args:
        config (dict): Configuration parameters.

    """

    def __init__(self, config):
        super().__init__(config)
        self._parse()
        self._oversample = 2

    @classmethod
    def defaults(cls):
        """(Class method) Return options supported by the operator and their defaults.

        This returns an ObjectConfig instance, and each entry should have a help
        string.

        Returns:
            (ObjectConfig): The options.

        """
        opts = ObjectConfig()

        opts.add("class", "toast.future_ops.SimNoise", "The class name")

        opts.add("API", 0, "(Internal interface version for this operator)")

        opts.add("out", None, "The name of the output signal")

        opts.add("realization", 0, "The realization index")

        opts.add("component", 0, "The component index")

        opts.add(
            "noise",
            "noise",
            "The observation key containing the noise model to use for simulations",
        )

        return opts

    def _parse(self):
        if self.config["realization"] < 0 or self.config["component"] < 0:
            raise RuntimeError("realization and component indices should be positive")
        if self.config["out"] is None:
            self.config["out"] = "SIGNAL"

    @function_timer
    def exec(self, data, detectors=None):
        """Generate noise timestreams.

        This iterates over all observations and detectors and generates
        the noise timestreams based on the noise object for the current
        observation.

        Args:
            data (toast.Data): The distributed data.
            detectors (list):  A list of detector names or indices.  If None, this
                indicates a list of all detectors.

        Raises:
            KeyError: If an observation does not contain the noise or output
                signal keys.

        """
        log = Logger.get()
        for obs in data.obs:
            # Get the detectors we are using for this observation
            dets = obs.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue

            # Unique observation ID
            obsindx = obs.UID

            # FIXME: we should unify naming of UID / id.
            telescope = obs.telescope.id

            # FIXME:  Every observation has a set of timestamps.  This global
            # offset is specified separately so that opens the possibility for
            # inconsistency.  Perhaps the global_offset should be made a property
            # of the Observation class?
            global_offset = 0
            if "global_offset" in obs:
                global_offset = obs["global_offset"]

            if self.config["noise"] not in obs:
                msg = "Observation does not contain noise key '{}'".format(
                    self.config["noise"]
                )
                log.error(msg)
                raise KeyError(msg)

            nse = obs[self.config["noise"]]

            # Eventually we'll redistribute, to allow long correlations...
            if obs.grid_size[1] != 1:
                msg = "Noise simulation for process grids with multiple ranks in the sample direction not implemented"
                log.error(msg)
                raise NotImplementedError(msg)

            # The previous code verified that a single process has whole
            # detectors within the observation.

            # Create output if it does not exist
            if self.config["out"] not in obs:
                obs.create_detector_data(
                    self.config["out"], shape=(obs.local_samples[1],), dtype=np.float64
                )

            (rate, dt, dt_min, dt_max, dt_std) = rate_from_times(obs.times)

            for key in nse.keys:
                # Check if noise matching this PSD key is needed
                weight = 0.0
                for det in dets:
                    weight += np.abs(nse.weight(det, key))
                if weight == 0:
                    continue

                # Simulate the noise matching this key
                nsedata = sim_noise_timestream(
                    self.config["realization"],
                    telescope,
                    self.config["component"],
                    obsindx,
                    nse.index(key),
                    rate,
                    obs.local_samples[0] + global_offset,
                    obs.local_samples[1],
                    self._oversample,
                    nse.freq(key),
                    nse.psd(key),
                )

                # Add the noise to all detectors that have nonzero weights
                for det in dets:
                    weight = nse.weight(det, key)
                    if weight == 0:
                        continue
                    obs.get_signal(keyname=self.config["out"])[
                        obs.local_samples[0] : obs.local_samples[0]
                        + obs.local_samples[1]
                    ] += (weight * nsedata)

            # Release the work space allocated in the FFT plan store.
            #
            # FIXME: the fact that we are doing this begs the question of why bother
            # using the plan store at all?  Only one plan per process, per FFT length
            # should be created.  The memory use of these plans should be small relative
            # to the other timestream memory use except in the case where:
            #
            #  1.  Each process only has a few detectors
            #  2.  There is a broad distribution of observation lengths.
            #
            # If we are in this regime frequently, we should just allocate / free each plan.
            store = FFTPlanReal1DStore.get()
            store.clear()

        return

    def finalize(self, data):
        """Perform any final operations / communication.

        This calls the finalize() method on all operators in sequence.

        Args:
            data (toast.Data):  The distributed data.

        Returns:
            None

        """
        return

    def requires(self):
        """List of Observation keys directly used by this Operator.
        """
        req = [self.config["noise"]]
        return req

    def provides(self):
        """List of Observation keys generated by this Operator.
        """
        prov = list()
        prov.append(self.config["out"])
        return prov

    def accelerators(self):
        """List of accelerators supported by this Operator.
        """
        return list()
