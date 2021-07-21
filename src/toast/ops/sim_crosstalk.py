# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import traitlets

from ..utils import Environment, Logger

from ..timing import function_timer, Timer

from ..noise_sim import AnalyticNoise

from .. import rng
from ..traits import trait_docs, Int, Unicode, Float, Bool, Instance, Quantity

from .operator import Operator


@trait_docs
class CrossTalk(Operator):
    """
    1.  The cross talk matrix can just be a dictionary of
    dictionaries of values (i.e. a sparse matrix) on every process.
    It does not need to be a dense matrix loaded from an HDF5 file.
    The calling code can create this however it likes.

    2. Each process has a DetectorData object representing the local data for some
    detectors and some timespan (e.g. obs.detdata["signal"]).
    It can make a copy of this and pass it to the next rank in the grid column.
    Each process receives a copy from the previous process in the column,
    accumulates to its local detectors, and passes it along.
    This continues until every process has accumulated the data
    from the other processes in the column.
    """
    # Class traits

    API = Int(0, help="Internal interface version for this operator")


    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    xtalk_mat_file = Unicode(
        None, allow_none=True, help="CrossTalk matrix dictionary of dictionaries"
    )

    detector_ordering=Unicode(
            "random", help="Initialize Crosstalk matrix with detector ordering: `random, gap,constant` default `random` ")

    xtalk_mat_value= Float(1. , help ="constant value to fill all the Cross Talk matrix elements, default=`1` ")

    realization = Int(0, help="integer to set a different random seed ")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def init_xtalk (self, data ):

        self.xtalk_mat={}

        for ob in data.obs:
            obsindx = ob.uid
            key1 = ( 65536 + self.realization   )
            key2 = obsindx
            counter1 = 0
            counter2 = 1234567
            Ndets= len(ob.telescope.focalplane.detectors )
            rngdata = rng.random(
                 Ndets ,
                 sampler="uniform_01",
                 key=(key1, key2),
                 counter=(counter1, counter2),
             )
            self.xtalk_mat[ob.name ]={}
            for k, det in  enumerate(ob.telescope.focalplane.detectors ):
                if self.detector_ordering == "random":
                    self.xtalk_mat [ob.name][det ]= rngdata[k]
                elif  self.detector_ordering == "constant":
                    self.xtalk_mat[ob.name][det] =self.xtalk_mat_value

        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()
        if self.xtalk_mat_file is None :
            self.init_xtalk(data )

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)

            if len(dets) == 0: continue
            comm = ob.comm
            rank = ob.comm_rank
            ob.detdata.ensure(self.det_data, detectors=dets)
            obsindx = ob.uid
            telescope = ob.telescope.uid
            focalplane = ob.telescope.focalplane
            import pdb; pdb.set_trace()
            for det in dets:
                detindx = focalplane[det]["uid"]

                # xtalk_mat[det][detdatacomm]
                # detdata_comm

                for  r in range(comm.size) :

                    if r== rank :continue
                    #ob.detdata[self.det_data][det] =



        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [
                self.boresight,
            ],
            "detdata": list(),
            "intervals": list(),
        }
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": [
                self.det_data,
            ],
        }
        return prov

    def _accelerators(self):
        return list()
