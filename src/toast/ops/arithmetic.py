# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

from ..utils import Logger

from ..mpi import MPI

from ..traits import trait_docs, Int, Unicode, List

from .operator import Operator


@trait_docs
class Add(Operator):
    """Add two detdata timestreams.

    The result is stored in the first detdata object.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    first = Unicode(None, allow_none=True, help="The first detdata object")

    second = Unicode(None, allow_none=True, help="The second detdata object")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.first is None:
            msg = "The first trait must be set before calling exec"
            log.error(msg)
            raise RuntimeError(msg)

        if self.second is None:
            msg = "The second trait must be set before calling exec"
            log.error(msg)
            raise RuntimeError(msg)

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue
            if self.first not in ob.detdata:
                msg = "The first detdata key '{}' does not exist in observation {}".format(
                    self.first, ob.name
                )
                log.error(msg)
                raise RuntimeError(msg)
            if self.second not in ob.detdata:
                msg = "The second detdata key '{}' does not exist in observation {}".format(
                    self.second, ob.name
                )
                log.error(msg)
                raise RuntimeError(msg)
            for d in dets:
                ob.detdata[self.first][d, :] += ob.detdata[self.second][d, :]

    def _finalize(self, data, **kwargs):
        return None

    def _requires(self):
        req = {"detdata": [self.first, self.second]}
        return req

    def _provides(self):
        prov = dict()
        return prov

    def _accelerators(self):
        # Eventually we can copy memory objects on devices...
        return list()


@trait_docs
class Subtract(Operator):
    """Subtract two detdata timestreams.

    The result is stored in the first detdata object.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    first = Unicode(None, allow_none=True, help="The first detdata object")

    second = Unicode(None, allow_none=True, help="The second detdata object")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        if self.first is None:
            msg = "The first trait must be set before calling exec"
            log.error(msg)
            raise RuntimeError(msg)

        if self.second is None:
            msg = "The second trait must be set before calling exec"
            log.error(msg)
            raise RuntimeError(msg)

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue
            if self.first not in ob.detdata:
                msg = "The first detdata key '{}' does not exist in observation {}".format(
                    self.first, ob.name
                )
                log.error(msg)
                raise RuntimeError(msg)
            if self.second not in ob.detdata:
                msg = "The second detdata key '{}' does not exist in observation {}".format(
                    self.second, ob.name
                )
                log.error(msg)
                raise RuntimeError(msg)
            for d in dets:
                ob.detdata[self.first][d, :] -= ob.detdata[self.second][d, :]

    def _finalize(self, data, **kwargs):
        return None

    def _requires(self):
        req = {"detdata": [self.first, self.second]}
        return req

    def _provides(self):
        prov = dict()
        return prov

    def _accelerators(self):
        # Eventually we can copy memory objects on devices...
        return list()
