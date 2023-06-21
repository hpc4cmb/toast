# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
import traitlets
from astropy import units as u

from .. import rng
from ..noise_sim import AnalyticNoise
from ..observation import default_values as defaults
from ..timing import Timer, function_timer
from ..traits import Float, Int, Unicode, Unit, trait_docs
from ..utils import Environment, Logger
from .operator import Operator


@function_timer
def read_xtalk_matrix(filename, data):
    """
    Read Xtalk matrix from disc.
    the  file to be in numpy binary format,
    i.e. `*.npz`.

    """

    matrix = np.load(filename)["matrix"]

    xtalk_mat = {}
    ob = data.obs[0]
    alldets = ob.telescope.focalplane.detectors
    Ndets = len(alldets)
    if Ndets > matrix.shape[0]:
        raise ValueError(
            f"Input Crosstalk matrix too small,  {matrix.shape} wrt  {Ndet} detectors to simulate."
        )

    for idet, det in enumerate(alldets):
        xtalk_mat[det] = {d: v for d, v in zip(alldets, matrix[idet, :])}
    return xtalk_mat


@function_timer
def init_xtalk_matrix(data, realization=0):
    """
    Initialize randomly a Xtalk matrix, with uniform values in [0,1].
    the matrix is the same for all the observations.
    """
    xtalk_mat = {}
    key1 = 65536 + realization
    counter1 = 0
    counter2 = 1234567
    ob = data.obs[0]
    alldets = ob.telescope.focalplane.detectors
    key2 = ob.session.uid
    Ndets = len(alldets)
    rngdata = rng.random(
        Ndets,
        sampler="uniform_01",
        key=(key1, key2),
        counter=(counter1, counter2),
    )
    #  assuming  we have a unique Xtalk matrix
    # for all the observations
    for det in alldets:
        xtalk_mat[det] = {d: v for d, v in zip(alldets, rngdata)}
        # Xtalk matrices have 0 diagonal
        xtalk_mat[det][det] = 0.0

    return xtalk_mat


@function_timer
def inject_error_in_xtalk_matrix(xtalk_mat, epsilon, realization=0):
    """
    Initialize randomly a Xtalk matrix, with uniform values in [0,1].
    the matrix is the same for all the observations.
    """
    key1 = 65536 + realization
    counter1 = 0
    counter2 = 1234567
    key2 = 9876
    xtalk_mat_bias = {}
    for det in xtalk_mat.keys():
        Ndets = len(xtalk_mat[det].keys())

        rngdata = rng.random(
            Ndets,
            sampler="uniform_01",
            key=(key1, key2),
            counter=(counter1, counter2),
        )
        xtalk_mat_bias[det] = {
            k[0]: (1 + rngdata[i] * epsilon) * k[1]
            for i, k in enumerate(xtalk_mat[det].items())
        }

    return xtalk_mat_bias


def invert_xtalk_mat(matdic):
    """
    To mitigate the Crosstalk we assume we have an  estimate
    of the crosstalk matrix,M. Then to correct for  Xtalk we evalute the
    inverse `Minv` defined as `Minv = inverse(1+M )`.


    """

    dets = list(matdic.keys())
    ndet = len(dets)
    M = np.zeros((ndet, ndet))

    for ii, det in enumerate(dets):
        M[ii, :] = np.array(list(matdic[det].values()))

        M[ii, ii] = 1
    Minv = np.linalg.inv(M)
    invdic = {}
    for ii, det in enumerate(dets):
        invdic[det] = {d: Minv[ii, jj] for jj, d in enumerate(matdic[det].keys())}

    return invdic


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
        defaults.det_data,
        allow_none=True,
        help="Observation detdata key for the timestream data",
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    xtalk_mat_file = Unicode(
        None, allow_none=True, help="CrossTalk matrix dictionary of dictionaries"
    )

    detector_ordering = Unicode(
        "random",
        help="Initialize Crosstalk matrix with detector ordering: `random, gap,constant` default `random` ",
    )

    realization = Int(0, help="integer to set a different random seed ")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def exec_roundrobin(self, data, detectors=None, **kwargs):
        env = Environment.get()
        log = Logger.get()
        if self.xtalk_mat_file is None:
            self.xtalk_mat = init_xtalk_matrix(data, realization=self.realization)

        for ob in data.obs:
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            Ndets = len(dets)
            if Ndets == 0:
                continue
            comm = ob.comm.comm_group
            rank = ob.comm.group_rank
            exists = ob.detdata.ensure(
                self.det_data, detectors=dets, create_units=self.det_data_units
            )
            telescope = ob.telescope.uid
            focalplane = ob.telescope.focalplane
            # we loop over all the procs except rank

            procs = np.arange(ob.comm.group_size)
            procs = procs[procs != rank]
            tmp = np.zeros_like(ob.detdata[self.det_data].data)
            for idet, det in enumerate(dets):
                xtalklist = list(self.xtalk_mat[det].keys())
                # we firstly xtalk local detectors in each rank
                intersect_local = np.intersect1d(
                    ob.detdata[self.det_data].detectors, xtalklist
                )
                ind1 = [xtalklist.index(k) for k in intersect_local]
                ind2 = [
                    ob.detdata[self.det_data].detectors.index(k)
                    for k in intersect_local
                ]

                xtalk_weights = np.array(
                    [self.xtalk_mat[det][kk] for kk in np.array(xtalklist)[ind1]]
                )

                tmp[idet] += np.dot(
                    xtalk_weights, ob.detdata[self.det_data].data[ind2, :]
                )
                # assert  old.var() !=   ob.detdata[self.det_data][det] .var()
                for ip in procs:
                    # send and recv data

                    comm.isend(
                        ob.detdata[self.det_data].detectors, dest=ip, tag=rank * 10 + ip
                    )
                    req = comm.irecv(source=ip, tag=ip * 10 + rank)
                    detlist = req.wait()
                    print(detlist, "detlist", rank, ip)
                    intersect = list(set(detlist).intersection(set(xtalklist)))
                    print(intersect, "intersect", rank, ip)
                    # intersect = np.intersect1d( detlist,xtalklist)
                    ## we make sure that we communicate the samples
                    # ONLY in case some of the  detectors sent by a rank  xtalking with det
                    # if intersect.size ==0: continue
                    if len(intersect) == 0:
                        continue

                    # define the indices of Xtalk coefficients and of detdata

                    ind1 = [xtalklist.index(k) for k in intersect]
                    ind2 = [detlist.index(k) for k in intersect]
                    xtalk_weights = np.array(
                        [self.xtalk_mat[det][kk] for kk in np.array(xtalklist)[ind1]]
                    )

                    # send and recv detdata

                    comm.isend(
                        ob.detdata[self.det_data].data, dest=ip, tag=rank * 10 + ip
                    )
                    # buf = bytearray(1<<20) # 1 MB buffer, make it larger if needed.
                    # req= comm.irecv( buf=buf,source=ip ,
                    req = comm.irecv(source=ip, tag=ip * 10 + rank)
                    #                    success,detdata = req.test()
                    detdata = req.wait()

                    tmp[idet] += np.dot(xtalk_weights, detdata[ind2])

            # before updating detdata samples we make sure
            # that all the send/receive have been performed
            comm.Barrier()
            for idet, det in enumerate(dets):
                ob.detdata[self.det_data][det] += tmp[idet]

        return

    @function_timer
    def _exec(self, data, **kwargs):
        env = Environment.get()
        log = Logger.get()
        ## Read the XTalk matrix from file or initialize it randomly

        if self.xtalk_mat_file is None:
            self.xtalk_mat = init_xtalk_matrix(data, realization=self.realization)
        else:
            self.xtalk_mat = read_xtalk_matrix(self.xtalk_mat_file, data)

        for ob in data.obs:
            # Get the detectors we are using for this observation
            comm = ob.comm.comm_group
            rank = ob.comm.group_rank
            # Detdata are usually distributed by detectors,
            # to crosstalk is more convenient to  redistribute them by time,
            # so that   each process has the samples from all detectors at a given
            # time stamp
            if ob.comm.group_size > 1:
                old_data_shape = ob.detdata[self.det_data].data.shape
                ob.redistribute(1, times=ob.shared["times"])
                # Now ob.local_detectors == ob.all_detectors and
                # the number of local samples is some small slice of the total
                new_data_shape = ob.detdata[self.det_data].data.shape
                assert old_data_shape != new_data_shape
                assert new_data_shape[0] == len(ob.all_detectors)

            # we store the crosstalked data into a temporary array
            tmp = np.zeros_like(ob.detdata[self.det_data].data)
            for idet, det in enumerate(ob.all_detectors):
                # for a given detector we assume that only a subset
                # of detectors can be crosstalking
                xtalklist = list(self.xtalk_mat[det].keys())
                intersect_local = np.intersect1d(ob.all_detectors, xtalklist)
                ind1 = [xtalklist.index(k) for k in intersect_local]
                # ind2 = [ ob.detdata[self.det_data].detectors .index(k)  for  k in intersect_local]
                ind2 = [ob.all_detectors.index(k) for k in intersect_local]
                xtalk_weights = np.array(
                    [self.xtalk_mat[det][kk] for kk in np.array(xtalklist)[ind1]]
                )
                tmp[idet] += np.dot(
                    xtalk_weights, ob.detdata[self.det_data].data[ind2, :]
                )

            for idet, det in enumerate(ob.all_detectors):
                ob.detdata[self.det_data][det] += tmp[idet]

            # We distribute the data back to the previous distribution
            if ob.comm.group_size > 1:
                ob.redistribute(ob.comm.group_size, times=ob.shared["times"])

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [
                self.boresight,
            ],
            "detdata": [self.det_data],
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


@trait_docs
class MitigateCrossTalk(Operator):
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

    realization = Int(0, help="integer to set a different random seed ")
    error_coefficients = Float(
        0, help="relative amplitude to simulate crosstalk errors on the inverse matrix "
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, **kwargs):
        env = Environment.get()
        log = Logger.get()

        ## Read the XTalk matrix from file or initialize it randomly
        if self.xtalk_mat_file is None:
            self.xtalk_mat = init_xtalk_matrix(data, realization=self.realization)
        else:
            self.xtalk_mat = read_xtalk_matrix(self.xtalk_mat_file, data)

        ## Inject an error to the matrix coefficients
        if self.error_coefficients:
            self.xtalk_mat = inject_error_in_xtalk_matrix(
                self.xtalk_mat, self.error_coefficients, realization=self.realization
            )
        # invert the Xtalk matrix (encoding the error )
        self.inv_xtalk_mat = invert_xtalk_mat(self.xtalk_mat)

        for ob in data.obs:
            # Get the detectors we are using for this observation
            comm = ob.comm.comm_group
            rank = ob.comm.group_rank
            # Redistribute data as in CrossTalk operator
            if ob.comm.group_size > 1:
                old_data_shape = ob.detdata[self.det_data].data.shape
                ob.redistribute(1, times=ob.shared["times"])
                new_data_shape = ob.detdata[self.det_data].data.shape
                assert new_data_shape[0] == len(ob.all_detectors)

            # we store the crosstalked data into a temporary array
            tmp = np.zeros_like(ob.detdata[self.det_data].data)

            for idet, det in enumerate(ob.all_detectors):
                # for a given detector only a subset
                # of detectors can be crosstalking

                xtalklist = list(self.xtalk_mat[det].keys())
                intersect_local = np.intersect1d(ob.all_detectors, xtalklist)
                ind1 = [xtalklist.index(k) for k in intersect_local]
                ind2 = [
                    ob.detdata[self.det_data].detectors.index(k)
                    for k in intersect_local
                ]

                xtalk_weights = np.array(
                    [self.inv_xtalk_mat[det][kk] for kk in np.array(xtalklist)[ind1]]
                )
                tmp[idet] += np.dot(
                    xtalk_weights, ob.detdata[self.det_data].data[ind2, :]
                )

            for idet, det in enumerate(ob.all_detectors):
                ob.detdata[self.det_data][det] = tmp[idet]
            # We distribute the data back to the previous distribution
            if ob.comm.group_size > 1:
                ob.redistribute(ob.comm.group_size, times=ob.shared["times"])

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
