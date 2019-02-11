# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""
sim_det_noise.py implements the noise simulation operator, OpSimNoise.

"""

import numpy as np

from ..op import Operator

from ..ctoast import sim_noise_sim_noise_timestream as sim_noise_timestream
from .. import timing as timing


class OpSimNoise(Operator):
    """
    Operator which generates noise timestreams.

    This passes through each observation and every process generates data
    for its assigned samples.  The dictionary for each observation should
    include a unique 'ID' used in the random number generation.  The
    observation dictionary can optionally include a 'global_offset' member
    that might be useful if you are splitting observations and want to
    enforce reproducibility of a given sample, even when using
    different-sized observations.

    Args:
        out (str): accumulate data to the cache with name <out>_<detector>.
            If the named cache objects do not exist, then they are created.
        realization (int): if simulating multiple realizations, the realization
            index.
        component (int): the component index to use for this noise simulation.
        noise (str): PSD key in the observation dictionary.

    """

    def __init__(self, out='noise', realization=0, component=0, noise='noise',
                 rate=None, altFFT=False):

        # We call the parent class constructor, which currently does nothing
        super().__init__()

        self._out = out
        self._oversample = 2
        self._realization = realization
        self._component = component
        self._noisekey = noise
        self._rate = rate
        self._altfft = altFFT

    def exec(self, data):
        """
        Generate noise timestreams.

        This iterates over all observations and detectors and generates
        the noise timestreams based on the noise object for the current
        observation.

        Args:
            data (toast.Data): The distributed data.

        Raises:
            KeyError: If an observation in data does not have noise
                object defined under given key.
            RuntimeError: If observations are not split into chunks.

        """
        autotimer = timing.auto_timer(type(self).__name__)
        for obs in data.obs:
            obsindx = 0
            if 'id' in obs:
                obsindx = obs['id']
            else:
                print("Warning: observation ID is not set, using zero!")

            telescope = 0
            if 'telescope' in obs:
                telescope = obs['telescope_id']

            global_offset = 0
            if 'global_offset' in obs:
                global_offset = obs['global_offset']

            tod = obs['tod']
            if self._noisekey in obs:
                nse = obs[self._noisekey]
            else:
                raise KeyError('Observation does not contain noise under '
                               '"{}"'.format(self._noisekey))
            if tod.local_chunks is None:
                raise RuntimeError('noise simulation for uniform distributed '
                                   'samples not implemented')

            # eventually we'll redistribute, to allow long correlations...

            if self._rate is None:
                times = tod.local_times()
            else:
                times = None

            # Iterate over each chunk.

            chunk_first = tod.local_samples[0]
            for curchunk in range(tod.local_chunks[1]):
                chunk_first += self.simulate_chunk(
                    tod=tod, nse=nse,
                    curchunk=curchunk, chunk_first=chunk_first,
                    obsindx=obsindx, times=times,
                    telescope=telescope, global_offset=global_offset)

        return

    def simulate_chunk(self, *, tod, nse, curchunk, chunk_first,
                       obsindx, times, telescope, global_offset):
        """
        Simulate one chunk of noise for all detectors.

        Args:
            tod (toast.tod.TOD): TOD object for the observation.
            nse (toast.tod.Noise): Noise object for the observation.
            curchunk (int): The local index of the chunk to simulate.
            chunk_first (int): First global sample index of the chunk.
            obsindx (int): Observation index for random number stream.
            times (int): Timestamps for effective sample rate.
            telescope (int): Telescope index for random number stream.
            global_offset (int): Global offset for random number stream.

        Returns:
            chunk_samp (int): Number of simulated samples

        """
        autotimer = timing.auto_timer(type(self).__name__)
        chunk_samp = tod.total_chunks[tod.local_chunks[0] + curchunk]
        local_offset = chunk_first - tod.local_samples[0]

        if self._rate is None:
            # compute effective sample rate
            rate = 1 / np.median(np.diff(
                times[local_offset : local_offset+chunk_samp]))
        else:
            rate = self._rate

        for key in nse.keys:
            # Check if noise matching this PSD key is needed
            weight = 0.
            for det in tod.local_dets:
                weight += np.abs(nse.weight(det, key))
            if weight == 0:
                continue

            # Simulate the noise matching this key
            #nsedata = sim_noise_timestream(
            #    self._realization, telescope, self._component, obsindx,
            #    nse.index(key), rate, chunk_first+global_offset, chunk_samp,
            #    self._oversample, nse.freq(key), nse.psd(key),
            #    self._altfft)[0]

            nsedata = sim_noise_timestream(
                self._realization, telescope, self._component, obsindx,
                nse.index(key), rate, chunk_first+global_offset, chunk_samp,
                self._oversample, nse.freq(key), nse.psd(key))

            # Add the noise to all detectors that have nonzero weights
            for det in tod.local_dets:
                weight = nse.weight(det, key)
                if weight == 0:
                    continue
                cachename = '{}_{}'.format(self._out, det)
                if tod.cache.exists(cachename):
                    ref = tod.cache.reference(cachename)
                else:
                    ref = tod.cache.create(cachename, np.float64,
                                           (tod.local_samples[1], ))
                ref[local_offset : local_offset+chunk_samp] += weight*nsedata
                del ref

        return chunk_samp
