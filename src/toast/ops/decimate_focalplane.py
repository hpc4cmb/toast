# Copyright (c) 2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


import numpy as np
import traitlets

from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import Environment, Logger, flagged_noise_fill, name_UID
from .operator import Operator


@trait_docs
class DecimateFocalplane(Operator):
    """An operator that disables all but every n:th detector on the focalplane."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    nskip = Int(
        10,
        help="The decimation factor. Only keep every nskip:th detector.",
    )

    detectors_per_pixel = Int(
        1,
        help="Assume that N consequtive detectors are in the same "
        "pixel and disable entire pixels at once. Use '2' for typical "
        "focalplanes and '3' or '6' for demodulated data.",
    )

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        if self.nskip == 1:
            # No flagging to do
            return
        
        log = Logger.get()
        wcomm = data.comm.comm_world
        comm = data.comm.comm_group
        timer0 = Timer()
        timer0.start()

        for ob in data.obs:
            if not ob.is_distributed_by_detector:
                msg = "Observation data must be distributed by detector, not samples"
                log.error(msg)
                raise RuntimeError(msg)

            local_dets = sorted(ob.select_local_detectors(detectors))
            good_dets = set(ob.select_local_detectors(detectors, flagmask=self.det_mask))
            ndet = len(local_dets)
            ndet_per_pix = self.detectors_per_pixel
            npix = ndet // ndet_per_pix
            nskip = self.nskip

            ntot = len(local_dets)
            ngood = len(good_dets)
            nbad = ntot - ngood
            ndecimated = 0

            decimate_flags = {}
            for ipix in range(npix):
                if ipix % nskip != 0:
                    # Flag this pixel
                    offset = ipix * ndet_per_pix
                    for idet in range(offset, offset + ndet_per_pix):
                        det = local_dets[idet]
                        decimate_flags[det] = self.det_mask
                        if det in good_dets:
                            ndecimated += 1
            ob.update_local_detector_flags(decimate_flags)

            if comm is not None:
                ntot = comm.allreduce(ntot)
                ngood = comm.allreduce(ngood)
                nbad = comm.allreduce(nbad)
                ndecimated = comm.allreduce(ndecimated)

            nleft = ngood - ndecimated
            frac = nleft / ntot
            log.debug_rank(
                f"Decimated {ndecimated} / {ngood} additional detectors in "
                f"{ob.name}.  There were already {nbad} / {ntot} flagged detectors. "
                f"Surviving fraction is {nleft} / {ntot} = {frac:.3f}.",
                comm=comm,
            )

        if detectors is None:
            log.info_rank(f"Applied {type(self).__name__} in", comm=wcomm, timer=timer0)
        else:
            log.debug_rank(f"Applied {type(self).__name__} in", comm=wcomm, timer=timer0)

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
            "intervals": list(),
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
            "intervals": list(),
        }
        return prov
