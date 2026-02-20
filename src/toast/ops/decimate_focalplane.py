# Copyright (c) 2025-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


import numpy as np
import traitlets

from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import Logger
from .operator import Operator


@trait_docs
class DecimateFocalplane(Operator):
    """An operator that disables all but every n:th detector on the focalplane."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    nskip = Int(
        10,
        help="The decimation factor. Only keep every nskip:th pixel.",
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

    pixel_property = Unicode(
        None,
        allow_none=True,
        help="Focalplane property for identifying different pixels. "
        "Overrides `detectors_per_pixel`",
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

            # The focalplane is identical on all processes in this group
            # so we assign and flag pixels locally

            fp = ob.telescope.focalplane

            if self.pixel_property is not None:
                # Assign detectors to pixels based on their focalplane properties
                if self.pixel_property not in fp.properties:
                    msg = f"{self.pixel_property} is not a property of the "
                    msg += f"focaplane in {ob.name}"
                    raise RuntimeError(msg)
                pixels = sorted(set(fp.detector_data[self.pixel_property]))
                keep_pixels = set(pixels[:: self.nskip])
                flag_pixels = set(pixels) - keep_pixels
                det_to_pixel = dict(
                    zip(fp.detectors, fp.detector_data[self.pixel_property])
                )
            else:
                # Use detector index to infer a dummy pixel name
                ndet_per_pix = self.detectors_per_pixel
                ndet = len(fp.detectors)
                npix = ndet // ndet_per_pix
                pixels = np.arange(npix, dtype=int)
                keep_pixels = set(pixels[:: self.nskip])
                flag_pixels = set(pixels) - keep_pixels
                det_to_pixel = {}
                for idet, det in enumerate(fp.detectors):
                    det_to_pixel[det] = idet // ndet_per_pix

            local_dets = ob.select_local_detectors(detectors)
            good_dets = set(
                ob.select_local_detectors(detectors, flagmask=self.det_mask)
            )
            ndet = len(local_dets)
            ntot = len(local_dets)
            ngood = len(good_dets)
            nbad = ntot - ngood
            ndecimated = 0

            decimate_flags = {}
            for det in local_dets:
                pix = det_to_pixel[det]
                if pix in flag_pixels:
                    # Flag this pixel
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
            log.debug_rank(
                f"Applied {type(self).__name__} in", comm=wcomm, timer=timer0
            )

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
