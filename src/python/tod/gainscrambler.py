# Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.

import re

import numpy as np

from ..op import Operator
from ..dist import Comm, Data
from .tod import TOD
from .. import rng as rng
from .. import timing as timing


class OpGainScrambler(Operator):
    """
    Operator which draws random gain errors from a given
    distribution and applies them to the specified detectors.

    Args:
        center (float):  Gain distribution center.
        sigma (float):  Gain distribution width.
        pattern (str):  Regex pattern to match against detector names.
            Only detectors that match the pattern are scrambled.
        name (str):  Name of the output signal cache object will be
            <name_in>_<detector>.  If the object exists, it is used as
            input.  Otherwise signal is read using the tod read method.
        realization (int): if simulating multiple realizations, the
            realization index.
        component (int): the component index to use for this noise
            simulation.
    """

    def __init__(self, center=1, sigma=1e-3, pattern=r'.*',
                 name=None, realization=0, component=234567):

        self._center = center
        self._sigma = sigma
        self._pattern = pattern
        self._name = name
        self._realization = realization
        self._component = component

        # We call the parent class constructor, which currently does nothing
        super().__init__()

    def exec(self, data):
        """
        Scramble the gains.

        Args:
            data (toast.Data): The distributed data.
        """
        autotimer = timing.auto_timer(type(self).__name__)
        # the two-level pytoast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world
        # the communicator within the group
        cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        crank = comm.comm_rank

        for obs in data.obs:
            obsindx = 0
            if 'id' in obs:
                obsindx = obs['id']
            else:
                print("Warning: observation ID is not set, using zero!")

            telescope = 0
            if 'telescope' in obs:
                telescope = obs['telescope']

            tod = obs['tod']

            pat = re.compile(self._pattern)

            for det in tod.local_dets:
                # Test the detector pattern

                if not pat.match(det):
                    continue

                detindx = tod.detindx[det]

                # Cache the output signal
                ref = tod.local_signal(det, self._name)

                """
                key1 = realization * 2^32 + telescope * 2^16 + component
                key2 = obsindx * 2^32 + detindx
                counter1 = currently unused (0)
                counter2 = currently unused (0)
                """

                key1 = self._realization * 4294967296 + telescope * 65536 \
                       + self._component
                key2 = obsindx * 4294967296 + detindx 
                counter1 = 0
                counter2 = 0

                rngdata = rng.random(1, sampler="gaussian", key=(key1, key2), 
                                     counter=(counter1, counter2))

                gain = self._center + rngdata[0] * self._sigma
                ref *= gain

                del ref

        return
