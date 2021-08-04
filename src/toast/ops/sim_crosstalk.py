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

@function_timer
def init_xtalk ( data , detectors=None
                    ,realization=0):

    xtalk_mat={}

    for ob in data.obs:
        obsindx = ob.uid
        key1 = ( 65536 +  realization  )
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
        dets = ob.select_local_detectors(detectors )
        alldets = ob.telescope.focalplane.detectors
        for   det in   dets :
            xtalk_mat [det ]= {d : v for d,v in zip(alldets ,rngdata)}
    return xtalk_mat


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
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()
        if self.xtalk_mat_file is None :
            self.xtalk_mat =init_xtalk(data , detectors ,
                            realization=self.realization     )

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            Ndets=len(dets)
            if Ndets == 0: continue
            comm = ob.comm
            rank = ob.comm_rank

            ob.detdata.ensure(self.det_data, detectors=dets)
            obsindx = ob.uid
            telescope = ob.telescope.uid
            focalplane = ob.telescope.focalplane
            #we loop over all the procs except rank

            procs= np.arange(comm.size )
            procs= procs[procs != rank]
            for det in dets:
                xtalklist = list( self.xtalk_mat[det].keys())
                #send and recv data
                for ip in procs :
                    if ip == comm.size -1 :
                        #if last rank
                        # sendrecv list of local dets
                        comm.send(ob.detdata[self.det_data].detectors ,
                                dest=0 , tag= rank*100+ip *Ndets + ip   )
                        detlist= comm.recv( source=ip -1,
                                    tag= rank*100+ip *Ndets + ip   )
                    elif ip==0:
                        #if 0 rank
                        comm.send(ob.detdata[self.det_data].detectors,
                            dest=ip+1, tag= rank*100+ip *Ndets + ip   )
                        detlist= comm.recv( source=comm.size -1,
                                tag= rank*100+ip *Ndets + ip   )
                    else :
                        # if any other rank
                         comm.send(ob.detdata[self.det_data].detectors ,
                                 dest=ip+1,
                                 tag= rank*100+ip *Ndets + ip  )
                         detlist= comm.recv( source=ip-1,
                          tag= rank*100+ip *Ndets + ip )

                    intersect = np.intersect1d( detlist,xtalklist)
                    ## we make sure that we communicate the samples
                    # ONLY in case some of the  detectors sent by a rank  xtalking with det
                    if intersect.size >0: continue
                    if ip == comm.size -1 :
                        #if last rank
                        # sendrecv samples
                        comm.send(ob.detdata[self.det_data].data,
                                dest=0 , tag= rank*100+ip *Ndets + ip   )
                        detdata= comm.recv( source=ip -1,
                                    tag= rank*100+ip *Ndets + ip   )
                    elif ip==0:
                        #if 0 rank
                        comm.send(ob.detdata[self.det_data].data,
                            dest=ip+1, tag= rank*100+ip *Ndets + ip   )
                        detdata= comm.recv( source=comm.size -1,
                                tag= rank*100+ip *Ndets + ip   )
                    else :
                        # if any other rank
                        comm.send(ob.detdata[self.det_data].data,
                                dest=ip+1,
                                tag= rank*100+ip *Ndets + ip  )
                        detdata= comm.recv( source=ip-1,
                         tag= rank*100+ip *Ndets + ip )

                    ind1 = np.where (xtalklist == intersect )
                    ind2 = np.where (detlist == intersect )
                    for i1,i2 in zip(ind1,ind2):
                        xtalk_det = xtalklist[i1]
                        ob.detdata[self.det_data][det].data += self.xtalk_mat[det][xtalk_det] * detdata[ind2]
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
