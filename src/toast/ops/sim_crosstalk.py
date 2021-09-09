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
def read_xtalk_matrix( filename, data , detectors=None ):
    """
    Read Xtalk matrix from disc.
    the  file to be in numpy binary format,
    i.e. `*.npz`.
    In case we want to run on a subset of detectors
    """

    matrix = np.load(filename)['matrix']

    xtalk_mat={}

    for ob in data.obs:
        Ndets= len(ob.telescope.focalplane.detectors )
        dets = ob.select_local_detectors(detectors )
        alldets = ob.telescope.focalplane.detectors
        for   idet,det in   enumerate(dets) :
            xtalk_mat [det ]= {d : v for d,v in zip(alldets ,matrix[idet,:])}
    return xtalk_mat

@function_timer
def init_xtalk_matrix ( data , detectors=None
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
            xtalk_mat[det][det]=0.
    return xtalk_mat

"""
from ..vis import set_matplotlib_backend
import os
import matplotlib.pyplot as plt

def plot_xtalk_mat(  matdic ):
    dets= list(matdic.keys() )
    ndet = len (dets )
    M = np.zeros((ndet,ndet ))
    for  ii,  det  in enumerate(dets  ):
        M[ii,:]= np.array(list (matdic[det].values() ))
    set_matplotlib_backend()
    outfile = os.path.join(  "xtalk_matrix.png")
    #plt.imshow(M,cmap='Greys'); plt.show ()
    #plt.savefig(outfile)
    #plt.close()
"""

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
    def exec_roundrobin(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()
        if self.xtalk_mat_file is None :
            self.xtalk_mat =init_xtalk_matrix(data , detectors ,
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

            procs= np.arange(ob.comm_size )
            #procs= np.arange(ob.comm.size )
            procs= procs[procs != rank]
            tmp= np.zeros_like(ob.detdata[self.det_data].data)
            for idet, det in enumerate(dets):

                xtalklist = list( self.xtalk_mat[det].keys())
                # we firstly xtalk local detectors in each rank
                intersect_local = np.intersect1d(ob.detdata[self.det_data].detectors ,xtalklist)
                ind1 =[ xtalklist.index(k ) for  k in intersect_local ]
                ind2 = [ ob.detdata[self.det_data].detectors .index(k)  for  k in intersect_local]

                xtalk_weights = np.array([self.xtalk_mat[det][kk] for kk in np.array(xtalklist)[ind1]])

                tmp[idet]  += np.dot( xtalk_weights,  ob.detdata[self.det_data].data[ind2,:])
                #assert  old.var() !=   ob.detdata[self.det_data][det] .var()
                for ip in procs :
                    #send and recv data

                    comm.isend(ob.detdata[self.det_data].detectors ,
                                dest=ip , tag= rank*10+   ip   )
                    req= comm.irecv(  source=ip ,
                                    tag=  ip*10+  rank    )
                    detlist=  req.wait()
                    print (detlist, "detlist" , rank , ip )
                    intersect =  list(set(detlist ).intersection(set(xtalklist )))
                    print(intersect, "intersect", rank , ip )
                    #intersect = np.intersect1d( detlist,xtalklist)
                    ## we make sure that we communicate the samples
                    # ONLY in case some of the  detectors sent by a rank  xtalking with det
                    #if intersect.size ==0: continue
                    if len(intersect)  ==0: continue

                    #define the indices of Xtalk coefficients and of detdata

                    ind1 = [ xtalklist.index(k ) for  k in  intersect]
                    ind2 =[ detlist.index(k)  for  k in intersect]
                    xtalk_weights = np.array([self.xtalk_mat[det][kk] for kk in np.array(xtalklist)[ind1]])

                    #send and recv detdata

                    comm.isend(ob.detdata[self.det_data].data ,
                                dest=ip , tag= rank*10 + ip   )
                    #buf = bytearray(1<<20) # 1 MB buffer, make it larger if needed.
                    #req= comm.irecv( buf=buf,source=ip ,
                    req= comm.irecv(  source=ip ,
                                    tag=  ip*10+  rank    )
#                    success,detdata = req.test()
                    detdata = req.wait()

                    tmp[idet]  += np.dot( xtalk_weights,  detdata[ind2])

            # before updating detdata samples we make sure
            #that all the send/receive have been performed
            comm.Barrier()
            for idet, det in enumerate(dets):

                ob.detdata[self.det_data][det]+=tmp[idet]



        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()
        if self.xtalk_mat_file is None :
            self.xtalk_mat =init_xtalk_matrix(data , detectors ,
                            realization=self.realization     )
        else:
            self.xtalk_mat= read_xtalk_matrix(self.xtalk_mat_file, data , detectors)

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

            # Detdata are usually distributed by detectors,
            # to crosstalk is more convenient to  redistribute them by time,
            # so that   each process has the samples from all detectors at a given
            # time stamp.

            ob.redistribute(1, times=ob.shared["times"])
            #Now ob.local_detectors == ob.all_detectors and
            # the number of local samples is some small slice of the total

            #assert ob.local_detectors == ob.all_detectors

            # we store the crosstalked data into a temporary array
            tmp= np.zeros_like(ob.detdata[self.det_data].data)
            for idet, det in enumerate(dets):
                # for a given detector only a subset
                # of detectors can be crosstalking
                xtalklist = list( self.xtalk_mat[det].keys())
                intersect_local = np.intersect1d(ob.local_detectors ,xtalklist)
                ind1 =[ xtalklist.index(k ) for  k in intersect_local ]
                ind2 = [ ob.detdata[self.det_data].detectors .index(k)  for  k in intersect_local]
                xtalk_weights = np.array([self.xtalk_mat[det][kk] for kk in np.array(xtalklist)[ind1]])
                tmp[idet]  += np.dot( xtalk_weights,  ob.detdata[self.det_data].data[ind2,:])

            for idet, det in enumerate(dets):
                ob.detdata[self.det_data][det]+=tmp[idet]

            # We distribute the data back to the previous distribution

            ob.redistribute(ob.comm_size , times=ob.shared["times"])


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
