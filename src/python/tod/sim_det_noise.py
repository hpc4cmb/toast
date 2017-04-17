# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import numpy as np

from .tod import TOD

from .noise import Noise

from ..op import Operator

from .tod_math import sim_noise_timestream


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
        """
        comm = data.comm

        for obs in data.obs:
            obsindx = 0
            if 'id' in obs:
                obsindx = obs['id']
            else:
                print("Warning: observation ID is not set, using zero!")

            telescope = 0
            if 'telescope' in obs:
                telescope = obs['telescope']

            global_offset = 0
            if 'global_offset' in obs:
                global_offset = obs['global_offset']

            tod = obs['tod']
            if self._noisekey in obs:
                nse = obs[self._noisekey]
            else:
                raise RuntimeError('Observation does not contain noise under '
                                   '"{}"'.format(self._noisekey))
            if tod.local_chunks is None:
                raise RuntimeError('noise simulation for uniform distributed '
                                   'samples not implemented')

            # eventually we'll redistribute, to allow long correlations...

            if self._rate is None:
                times = tod.read_times(local_start=0, n=tod.local_samples[1])

            # Iterate over each chunk.

            tod_first = tod.local_samples[0]
            chunk_first = tod_first

            for curchunk in range(tod.local_chunks[1]):
                abschunk = tod.local_chunks[0] + curchunk
                chunk_samp = tod.total_chunks[abschunk]
                local_offset = chunk_first - tod_first

                if self._rate is None:
                    # compute effective sample rate
                    dt = np.median(np.diff(
                            times[local_offset:local_offset+chunk_samp]))
                    rate = 1.0 / dt
                else:
                    rate = self._rate

                idet = 0
                for det in tod.local_dets:

                    detindx = tod.detindx[det]

                    (nsedata, freq, psd) = sim_noise_timestream(
                        self._realization, telescope, self._component, obsindx,
                        detindx, rate, chunk_first + global_offset, chunk_samp,
                        self._oversample, nse.freq(det), nse.psd(det), 
                        self._altfft)

                    # write to cache

                    cachename = "{}_{}".format(self._out, det)

                    ref = None
                    if tod.cache.exists(cachename):
                        ref = tod.cache.reference(cachename)
                    else:
                        ref = tod.cache.create(cachename, np.float64,
                                    (tod.local_samples[1],))

                    ref[local_offset:local_offset+chunk_samp] += nsedata
                    del ref

                    idet += 1

                chunk_first += chunk_samp

        return

