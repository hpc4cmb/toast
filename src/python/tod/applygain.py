# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import re

from toast.op import Operator

import toast.rng as rng
import toast.timing as timing
from toast.tod.tod_math import calibrate


class OpApplyGain(Operator):
    """
    Operator which applies gains to timelines

    Args:
        gain (dict):  Dictionary, keys are channel names, values
            are dictionaries with keys "TIME" and "GAIN", values are
            arrays
        name (str):  Name of the output signal cache object will be
            <name_in>_<detector>.  If the object exists, it is used as
            input.  Otherwise signal is read using the tod read method.
    """

    def __init__(self, gain, name=None):

        self._gain = gain
        self._name = name

        # We call the parent class constructor, which currently does nothing
        super().__init__()

    def exec(self, data):
        """
        Apply the gains.

        Args:
            data (toast.Data): The distributed data.
        """
        autotimer = timing.auto_timer(type(self).__name__)

        for obs in data.obs:

            tod = obs['tod']

            for det in tod.local_dets:

                # Cache the output signal
                ref = tod.local_signal(det, self._name)
                obs_times = tod.read_times()

                calibrate(obs_times, ref, self._gain[det.upper()]["TIME"], self._gain[det.upper()]["GAIN"], order=0, inplace=True)

                assert np.isnan(ref).sum() == 0, "The signal timestream includes NaN"

                del ref

        return
