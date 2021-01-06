# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool, Instance, Float

from ..data import Data

from .template import Template


@trait_docs
class Fourier2D(Template):
    """This class represents atmospheric fluctuations in front of the focalplane
    as 2D Fourier modes.

    """

    # Notes:  The TraitConfig base class defines a "name" attribute.  The Template
    # class (derived from TraitConfig) defines the following traits already:
    #    data             : The Data instance we are working with
    #    view             : The timestream view we are using
    #    det_data         : The detector data key with the timestreams
    #    det_flags        : Optional detector flags
    #    det_flag_mask    : Bit mask for detector flags
    #    shared_flags     : Optional shared flags
    #    shared_flag_mask : Bit mask for shared flags
    #

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, new_data):
        return

    def _zeros(self):
        raise NotImplementedError("Derived class must implement _zeros()")

    def _add_to_signal(self, detector, amplitudes):
        raise NotImplementedError("Derived class must implement _add_to_signal()")

    def _project_signal(self, detector, amplitudes):
        raise NotImplementedError("Derived class must implement _project_signal()")

    def _add_prior(self, amplitudes_in, amplitudes_out):
        # Not all Templates implement the prior
        return

    def _apply_precond(self, amplitudes_in, amplitudes_out):
        raise NotImplementedError("Derived class must implement _apply_precond()")
