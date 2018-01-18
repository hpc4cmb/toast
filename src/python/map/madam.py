# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


from ..mpi import MPI

import ctypes as ct
from ctypes.util import find_library

import numpy as np
import numpy.ctypeslib as npc

import healpy as hp

from ..dist import Comm, Data
from ..op import Operator
from ..tod import TOD
from ..cache import Cache

from .. import timing as timing

libmadam = None

try:
    libmadam = ct.CDLL('libmadam.so')
except:
    path = find_library('madam')
    if path is not None:
        libmadam = ct.CDLL(path)

if libmadam is not None:
    libmadam.destripe.restype = None
    libmadam.destripe.argtypes = [
        ct.c_int,
        ct.c_char_p,
        ct.c_long,
        ct.c_char_p,
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ct.c_long,
        ct.c_long,
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        npc.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ct.c_long,
        npc.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
        npc.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
        ct.c_long,
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ct.c_long,
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ct.c_long,
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    libmadam.destripe_with_cache.restype = None
    libmadam.destripe_with_cache.argtypes = [
        ct.c_int,
        ct.c_long,
        ct.c_long,
        ct.c_long,
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        npc.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ct.c_char_p,
    ]
    libmadam.clear_caches.restype = None
    libmadam.clear_caches.argtypes = []

# Some keys may be defined multiple times in the Madam parameter files.
# Assume that such entries are aggregated into a list in a parameter
# dictionary

repeated_keys = ['detset', 'detset_nopol', 'survey']


class OpMadam(Operator):
    """
    Operator which passes data to libmadam for map-making.

    Args:
        params (dictionary): parameters to pass to madam.
        detweights (dictionary): individual noise weights to use for each
            detector.
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        pixels_nested (bool): Set to False if the pixel numbers are in
            ring ordering. Default is True.
        weights (str): the name of the cache object (<weights>_<detector>)
            containing the pointing weights to use.
        name (str): the name of the cache object (<name>_<detector>) to
            use for the detector timestream.  If None, use the TOD.
        name_out (str): the name of the cache object (<name>_<detector>)
            to use to output destriped detector timestream.
            No output if None.
        flag_name (str): the name of the cache object
            (<flag_name>_<detector>) to use for the detector flags.
            If None, use the TOD.
        flag_mask (int): the integer bit mask (0-255) that should be
            used with the detector flags in a bitwise AND.
        common_flag_name (str): the name of the cache object
            to use for the common flags.  If None, use the TOD.
        common_flag_mask (int): the integer bit mask (0-255) that should
            be used with the common flags in a bitwise AND.
        apply_flags (bool): whether to apply flags to the pixel numbers.
        purge (bool): if True, clear any cached data that is copied into
            the Madam buffers.
        purge_tod (bool): if True, clear any cached signal that is
            copied into the Madam buffers.
        purge_pixels (bool): if True, clear any cached pixels that are
            copied into the Madam buffers.
        purge_weights (bool): if True, clear any cached weights that are
            copied into the Madam buffers.
        purge_flags (bool): if True, clear any cached flags that are
            copied into the Madam buffers.
        dets (iterable):  List of detectors to map. If left as None, all
            available detectors are mapped.
        mcmode (bool): If true, the operator is constructed in
             Monte Carlo mode and Madam will cache auxiliary information
             such as pixel matrices and noise filter.
        noise (str):  Keyword to use when retrieving the noise object
             from the observation.
    """

    def __init__(self, params={}, detweights=None,
                 pixels='pixels', pixels_nested=True, weights='weights',
                 name=None, name_out=None, flag_name=None, flag_mask=255,
                 common_flag_name=None, common_flag_mask=255,
                 apply_flags=True, purge=False, dets=None, mcmode=False,
                 purge_tod=False, purge_pixels=False, purge_weights=False,
                 purge_flags=False, noise='noise', intervals='intervals'):

        # We call the parent class constructor, which currently does nothing
        super().__init__()
        # madam uses time-based distribution
        self._name = name
        self._name_out = name_out
        self._flag_name = flag_name
        self._flag_mask = flag_mask
        self._common_flag_name = common_flag_name
        self._common_flag_mask = common_flag_mask
        self._pixels = pixels
        self._pixels_nested = pixels_nested
        self._weights = weights
        self._detw = detweights
        self._purge = purge
        if self._purge:
            self._purge_tod = True
            self._purge_pixels = True
            self._purge_weights = True
            self._purge_flags = True
        else:
            self._purge_tod = purge_tod
            self._purge_pixels = purge_pixels
            self._purge_weights = purge_weights
            self._purge_flags = purge_flags
        self._apply_flags = apply_flags
        self.params = params
        if dets is not None:
            self._dets = set(dets)
        else:
            self._dets = None
        self._mcmode = mcmode
        if mcmode:
            self.params['mcmode'] = True
        else:
            self.params['mcmode'] = False
        if self._name_out is not None:
            self.params['write_tod'] = True
        else:
            self.params['write_tod'] = False
        self._cached = False
        self._noisekey = noise
        self._intervals = intervals
        self._cache = Cache()
        self._madam_timestamps = None
        self._madam_pixels = None
        self._madam_pixweights = None
        self._madam_signal = None

    def __del__(self):
        if self._cached:
            libmadam.clear_caches()
            self._cached = False

    @property
    def available(self):
        """
        (bool): True if libmadam is found in the library search path.
        """
        return (libmadam is not None)

    def _dict2parstring(self, d):
        s = ''
        for key, value in d.items():
            if key in repeated_keys:
                for separate_value in value:
                    s += '{} = {};'.format(key, separate_value)
            else:
                s += '{} = {};'.format(key, value)
        return s

    def _dets2detstring(self, dets):
        s = ''
        for d in dets:
            s += '{};'.format(d)
        return s

    def exec(self, data, comm=None):
        """
        Copy data to Madam-compatible buffers and make a map.

        Args:
            data (toast.Data): The distributed data.
        """
        if libmadam is None:
            raise RuntimeError("Cannot find libmadam")

        if len(data.obs) == 0:
            raise RuntimeError('OpMadam requires every supplied data object to '
                               'contain at least one observation')

        auto_timer = timing.auto_timer(type(self).__name__)

        if comm is None:
            # Just use COMM_WORLD
            comm = data.comm.comm_world

        (parstring, detstring, nsamp, ndet, nnz, nnz_full, nnz_stride, periods,
         obs_period_ranges, psdfreqs, detectors, nside) = self._prepare(data,
                                                                        comm)

        psdinfo, pixels_dtype, weight_dtype = self._stage_data(
            data, comm, nsamp, ndet, nnz, nnz_full, nnz_stride,
            obs_period_ranges, psdfreqs, detectors, nside)

        self._destripe(comm, parstring, ndet, detstring, nsamp, nnz, periods,
                       psdinfo)

        self._unstage_data(comm, data, nsamp, nnz, nnz_full,
                           obs_period_ranges, detectors, pixels_dtype, nside,
                           weight_dtype)

        return

    def _destripe(self, comm, parstring, ndet, detstring, nsamp, nnz, periods,
                  psdinfo):
        """ Destripe the buffered data

        """
        auto_timer = timing.auto_timer(type(self).__name__)
        fcomm = comm.py2f()
        if self._cached:
            # destripe
            outpath = ''
            if 'path_output' in self.params:
                outpath = self.params['path_output']
            outpath = outpath.encode('ascii')
            libmadam.destripe_with_cache(
                fcomm, ndet, nsamp, nnz, self._madam_timestamps,
                self._madam_pixels, self._madam_pixweights, self._madam_signal,
                outpath)
        else:
            (detweights, npsd, npsdtot, psdstarts, npsdbin, psdfreqs,
             npsdval, psdvals) = psdinfo

            # destripe
            libmadam.destripe(
                fcomm, parstring.encode(), ndet, detstring.encode(), detweights,
                nsamp, nnz, self._madam_timestamps, self._madam_pixels,
                self._madam_pixweights, self._madam_signal, len(periods), periods,
                npsd, npsdtot, psdstarts, npsdbin, psdfreqs, npsdval, psdvals)

            if self._mcmode:
                self._cached = True
        return

    def _count_samples(self, data):
        """ Loop over the observations and count the number of samples.

        """
        if len(data.obs) != 1:
            nsamp = 0
            tod0 = data.obs[0]['tod']
            detectors0 = tod0.local_dets
            for obs in data.obs:
                tod = obs['tod']
                # For the moment, we require that all observations have
                # the same set of detectors
                detectors = tod.local_dets
                dets_are_same = True
                if len(detectors0) != len(detectors):
                    dets_are_same = False
                else:
                    for det1, det2 in zip(detectors0, detectors):
                        if det1 != det2:
                            dets_are_same = False
                            break
                if not dets_are_same:
                    raise RuntimeError(
                        'When calling Madam, all TOD assigned to a process '
                        'must have the same local detectors.')
                nsamp += tod.local_samples[1]
        else:
            tod = data.obs[0]['tod']
            nsamp = tod.local_samples[1]
        return nsamp

    def _get_period_ranges(self, comm, data, detectors, nsamp):
        """ Collect the ranges of every observation.

        """
        # Discard intervals that are too short to fit a baseline
        if 'basis_order' in self.params:
            norder = int(self.params['basis_order']) + 1
        else:
            norder = 1

        psdfreqs = None
        period_lengths = []
        obs_period_ranges = []

        for obs in data.obs:
            tod = obs['tod']
            # Check that all noise objects have the same binning
            if self._noisekey in obs.keys():
                nse = obs[self._noisekey]
                if nse is not None:
                    if psdfreqs is None:
                        psdfreqs = nse.freq(detectors[0]).astype(
                            np.float64).copy()
                        npsdbin = len(psdfreqs)
                    for idet, det in enumerate(detectors):
                        check_psdfreqs = nse.freq(det)
                        if not np.allclose(psdfreqs, check_psdfreqs):
                            raise RuntimeError(
                                'All PSDs passed to Madam must have'
                                ' the same frequency binning.')
            # Collect the valid intervals for this observation
            nlocal = tod.local_samples[1]
            period_ranges = []
            if self._intervals in obs:
                intervals = obs[self._intervals]
            else:
                intervals = None
            local_intervals = tod.local_intervals(intervals)

            local_offset = tod.local_samples[0]
            local_nsamp = tod.local_samples[1]
            for ival in local_intervals:
                local_start = ival.first
                local_stop = ival.last
                if local_stop - local_start + 1 < norder:
                    continue
                period_lengths.append(local_stop - local_start + 1)
                period_ranges.append((local_start, local_stop + 1))
            obs_period_ranges.append(period_ranges)

        nsamp_tot_full = comm.allreduce(nsamp, op=MPI.SUM)
        nperiod = len(period_lengths)
        period_lengths = np.array(period_lengths, dtype=np.int64)
        nsamp = np.sum(period_lengths, dtype=np.int64)
        nsamp_tot = comm.allreduce(nsamp, op=MPI.SUM)
        if nsamp_tot == 0:
            raise RuntimeError(
                'No samples in valid intervals: nsamp_tot_full = {}, '
                'nsamp_tot = {}'.format(nsamp_tot_full, nsamp_tot))
        if comm.rank == 0:
            print('OpMadam: {:.2f} % of samples are included in valid '
                  'intervals.'.format(nsamp_tot*100./nsamp_tot_full))

        # Madam expects starting indices, not period lengths
        periods = np.zeros(nperiod, dtype=np.int64)
        for i, n in enumerate(period_lengths[:-1]):
            periods[i+1] = periods[i] + n

        return obs_period_ranges, psdfreqs, periods

    def _prepare(self, data, comm):
        """ Examine the data object.

        """
        auto_timer = timing.auto_timer(type(self).__name__)

        nsamp = self._count_samples(data)

        # Determine the detectors and the pointing matrix non-zeros
        # from the first observation. Madam will expect these to remain
        # unchanged across observations.

        tod = data.obs[0]['tod']

        if self._dets is None:
            detectors = tod.local_dets
        else:
            detectors = [det for det in tod.local_dets
                         if det in self._dets]
        ndet = len(detectors)
        detstring = self._dets2detstring(detectors)

        # to get the number of Non-zero pointing weights per pixel,
        # we use the fact that for Madam, all processes have all detectors
        # for some slice of time.  So we can get this information from the
        # shape of the data from the first detector

        nnzname = "{}_{}".format(self._weights, detectors[0])
        nnz_full = tod.cache.reference(nnzname).shape[1]

        if 'temperature_only' in self.params \
           and self.params['temperature_only'] in [
               'T', 'True', 'TRUE', 'true', True]:
            if nnz_full not in [1, 3]:
                raise RuntimeError(
                    'OpMadam: Don\'t know how to make a temperature map '
                    'with nnz={}'.format(nnz_full))
            nnz = 1
            nnz_stride = nnz_full
        else:
            nnz = nnz_full
            nnz_stride = 1

        if 'nside_map' not in self.params:
            raise RuntimeError(
                'OpMadam: "nside_map" must be set in the parameter dictionary')
        nside = int(self.params['nside_map'])

        parstring = self._dict2parstring(self.params)

        # Inspect the valid intervals across all observations to
        # determine the number of samples per detector

        obs_period_ranges, psdfreqs, periods \
            = self._get_period_ranges(comm, data, detectors, nsamp)

        return (parstring, detstring, nsamp, ndet, nnz, nnz_full, nnz_stride,
                periods, obs_period_ranges, psdfreqs, detectors, nside)

    def _stage_time(self, data, detectors, nsamp, obs_period_ranges):
        """ Stage the timestamps and use them to build PSD inputs.

        """
        auto_timer = timing.auto_timer(type(self).__name__)
        self._madam_timestamps = self._cache.create(
            'timestamps', np.float64, (nsamp, ))

        global_offset = 0
        time_offset = 0
        psds = {}
        for iobs, obs in enumerate(data.obs):
            tod = obs['tod']
            nlocal = tod.local_samples[1]
            period_ranges = obs_period_ranges[iobs]

            # Collect the timestamps for the valid intervals
            timestamps = tod.local_times()
            # Translate the time stamps to be monotonous
            timestamps -= timestamps[0] - time_offset
            time_offset = timestamps[-1] + 1

            offset = global_offset
            for istart, istop in period_ranges:
                nn = istop - istart
                ind = slice(offset, offset+nn)
                self._madam_timestamps[ind] = timestamps[istart:istop]
                offset += nn

            # get the noise object for this observation and create new
            # entries in the dictionary when the PSD actually changes
            if self._noisekey in obs.keys():
                nse = obs[self._noisekey]
                if 'noise_scale' in obs:
                    noise_scale = obs['noise_scale']
                else:
                    noise_scale = 1
                if nse is not None:
                    for idet, det in enumerate(detectors):
                        psd = nse.psd(det) * noise_scale**2
                        if det not in psds:
                            psds[det] = [(0, psd)]
                        else:
                            if not np.allclose(psds[det][-1][1], psd):
                                psds[det] += [(timestamps[0], psd)]

            global_offset = offset

        return psds

    def _stage_signal(self, data, detectors, nsamp, ndet,
                                  obs_period_ranges):
        """ Stage signal

        """
        auto_timer = timing.auto_timer(type(self).__name__)
        self._madam_signal = self._cache.create(
            'signal', np.float64, (nsamp*ndet, ))

        global_offset = 0
        time_offset = 0
        for iobs, obs in enumerate(data.obs):
            tod = obs['tod']
            nlocal = tod.local_samples[1]
            period_ranges = obs_period_ranges[iobs]

            for idet, det in enumerate(detectors):
                # Get the signal.
                signal = tod.local_signal(det, self._name)
                offset = global_offset
                for istart, istop in period_ranges:
                    nn = istop - istart
                    dslice = slice(idet*nsamp + offset,
                                   idet*nsamp + offset + nn)
                    self._madam_signal[dslice] = signal[istart:istop]
                    offset += nn

                del signal

            for idet, det in enumerate(detectors):
                if self._name is not None and (
                        self._purge_tod or self._name == self._name_out):
                    cachename = "{}_{}".format(self._name, det)
                    tod.cache.clear(pattern=cachename)

            global_offset = offset

        return

    def _stage_pixels(self, data, detectors, nsamp, ndet, obs_period_ranges):
        """ Stage pixels

        """
        auto_timer = timing.auto_timer(type(self).__name__)
        self._madam_pixels = self._cache.create(
            'pixels', np.int64, (nsamp*ndet, ))

        global_offset = 0
        time_offset = 0
        for iobs, obs in enumerate(data.obs):
            tod = obs['tod']
            nlocal = tod.local_samples[1]
            period_ranges = obs_period_ranges[iobs]

            commonflags = None
            for idet, det in enumerate(detectors):
                # Optionally get the flags, otherwise they are
                # assumed to have been applied to the pixel numbers.

                if self._apply_flags:
                    detflags = tod.local_flags(det, self._flag_name)
                    commonflags = tod.local_common_flags(
                        self._common_flag_name)
                    flags = np.logical_or(
                        (detflags & self._flag_mask) != 0,
                        (commonflags & self._common_flag_mask) != 0)
                    del detflags

                # get the pixels for the valid intervals from the cache

                pixelsname = "{}_{}".format(self._pixels, det)
                pixels = tod.cache.reference(pixelsname)
                pixels_dtype = pixels.dtype

                if not self._pixels_nested:
                    # Madam expects the pixels to be in nested ordering
                    pixels = pixels.copy()
                    good = pixels >= 0
                    pixels[good] = hp.ring2nest(nside, pixels[good])

                if self._apply_flags:
                    pixels = pixels.copy()
                    pixels[flags] = -1

                offset = global_offset
                for istart, istop in period_ranges:
                    nn = istop - istart
                    dslice = slice(idet*nsamp + offset,
                                   idet*nsamp + offset + nn)
                    self._madam_pixels[dslice] = pixels[istart:istop]
                    offset += nn

                del pixels
                if self._apply_flags:
                    del flags

            # Always purge the pixels but restore them from the Madam
            # buffers when purge_pixels=False
            for idet, det in enumerate(detectors):
                pixelsname = "{}_{}".format(self._pixels, det)
                tod.cache.clear(pattern=pixelsname)
                if self._name is not None and (
                        self._purge_tod or self._name == self._name_out):
                    cachename = "{}_{}".format(self._name, det)
                    tod.cache.clear(pattern=cachename)
                if self._purge_flags and self._flag_name is not None:
                    cacheflagname = "{}_{}".format(
                        self._flag_name, det)
                    tod.cache.clear(pattern=cacheflagname)

            del commonflags
            if self._purge_flags and self._common_flag_name is not None:
                tod.cache.clear(pattern=self._common_flag_name)
            global_offset = offset

        return pixels_dtype

    def _stage_pixweights(self, data, detectors, nsamp, ndet, nnz, nnz_full,
                          nnz_stride, obs_period_ranges):
        """Now collect the pixel weights

        """
        auto_timer = timing.auto_timer(type(self).__name__)

        self._madam_pixweights = self._cache.create(
            'pixweights', np.float64, (nsamp*ndet*nnz, ))

        global_offset = 0
        for iobs, obs in enumerate(data.obs):
            tod = obs['tod']
            nlocal = tod.local_samples[1]
            period_ranges = obs_period_ranges[iobs]
            for idet, det in enumerate(detectors):
                # get the pixels and weights for the valid intervals
                # from the cache
                weightsname = "{}_{}".format(self._weights, det)
                weights = tod.cache.reference(weightsname)
                weight_dtype = weights.dtype
                offset = global_offset
                for istart, istop in period_ranges:
                    nn = istop - istart
                    dwslice = slice((idet*nsamp+offset) * nnz,
                                    (idet*nsamp+offset+nn) * nnz)
                    self._madam_pixweights[dwslice] \
                        = weights[istart:istop].flatten()[::nnz_stride]
                    offset += nn
                del weights
            # Purge the weights but restore them from the Madam
            # buffers when purge_weights=False.
            # Handle special case when Madam only stores a subset of
            # the weights.
            if not self._purge_weights and (nnz != nnz_full):
                pass
            else:
                for idet, det in enumerate(detectors):
                    # get the pixels and weights for the valid intervals
                    # from the cache
                    weightsname = "{}_{}".format(self._weights, det)
                    tod.cache.clear(pattern=weightsname)

            global_offset = offset

        return

    def _stage_data(self, data, comm, nsamp, ndet, nnz, nnz_full, nnz_stride,
                    obs_period_ranges, psdfreqs, detectors, nside):
        """ create madam-compatible buffers

        Collect the TOD into Madam buffers. Process pixel weights
        Separate from the rest to reduce the memory high water mark
        When the user has set purge=True

        Moving data between toast and Madam buffers has an overhead.
        We perform the operation in a staggered fashion to have the
        overhead only once per node.

        """
        auto_timer = timing.auto_timer(type(self).__name__)

        nodecomm = comm.Split_type(MPI.COMM_TYPE_SHARED, comm.rank)
        nread = nodecomm.size
        nread = comm.allreduce(nread, MPI.MAX)

        for iread in range(nread):
            comm.Barrier()
            if nodecomm.rank % nread != iread:
                continue
            psds = self._stage_time(data, detectors, nsamp, obs_period_ranges)
            self._stage_signal(data, detectors, nsamp, ndet, obs_period_ranges)
            pixels_dtype = self._stage_pixels(data, detectors, nsamp, ndet,
                                              obs_period_ranges)
            weight_dtype = self._stage_pixweights(
                data, detectors, nsamp, ndet, nnz, nnz_full, nnz_stride,
                obs_period_ranges)
        del nodecomm

        # detweights is either a dictionary of weights specified at
        # construction time, or else we use uniform weighting.
        detw = {}
        if self._detw is None:
            for idet, det in enumerate(detectors):
                detw[det] = 1.0
        else:
            detw = self._detw

        detweights = np.zeros(ndet, dtype=np.float64)
        for idet, det in enumerate(detectors):
            detweights[idet] = detw[det]

        if len(psds) > 0:
            npsdbin = len(psdfreqs)

            npsd = np.zeros(ndet, dtype=np.int64)
            psdstarts = []
            psdvals = []
            for idet, det in enumerate(detectors):
                if det not in psds:
                    raise RuntimeError('Every detector must have at least '
                                       'one PSD')
                psdlist = psds[det]
                npsd[idet] = len(psdlist)
                for psdstart, psd in psdlist:
                    psdstarts.append(psdstart)
                    psdvals.append(psd)
            npsdtot = np.sum(npsd)
            psdstarts = np.array(psdstarts, dtype=np.float64)
            psdvals = np.hstack(psdvals).astype(np.float64)
            npsdval = psdvals.size
        else:
            npsd = np.ones(ndet, dtype=np.int64)
            npsdtot = np.sum(npsd)
            psdstarts = np.zeros(npsdtot)
            npsdbin = 10
            fsample = 10.
            psdfreqs = np.arange(npsdbin) * fsample / npsdbin
            npsdval = npsdbin * npsdtot
            psdvals = np.ones(npsdval)
        psdinfo = (detweights, npsd, npsdtot, psdstarts, npsdbin, psdfreqs,
                   npsdval, psdvals)

        return psdinfo, pixels_dtype, weight_dtype

    def _unstage_data(self, comm, data, nsamp, nnz, nnz_full,
                      obs_period_ranges, detectors, pixels_dtype, nside,
                      weight_dtype):
        """ Clear Madam buffers, restore pointing into TOAST caches
        and cache the destriped signal.

        """
        auto_timer = timing.auto_timer(type(self).__name__)
        self._madam_timestamps = None
        self._cache.destroy('timestamps')

        if self._name_out is not None:
            global_offset = 0
            for obs, period_ranges in zip(data.obs, obs_period_ranges):
                tod = obs['tod']
                nlocal = tod.local_samples[1]
                for idet, det in enumerate(detectors):
                    signal = np.ones(nlocal) * np.nan
                    offset = global_offset
                    for istart, istop in period_ranges:
                        nn = istop - istart
                        dslice = slice(idet*nsamp + offset,
                                       idet*nsamp + offset + nn)
                        signal[istart:istop] = self._madam_signal[dslice]
                        offset += nn
                    cachename = "{}_{}".format(self._name_out, det)
                    tod.cache.put(cachename, signal, replace=True)
                global_offset = offset
        self._madam_signal = None
        self._cache.destroy('signal')

        if not self._purge_pixels:
            # restore the pixels from the Madam buffers
            global_offset = 0
            for obs, period_ranges in zip(data.obs, obs_period_ranges):
                tod = obs['tod']
                nlocal = tod.local_samples[1]
                for idet, det in enumerate(detectors):
                    pixels = -np.ones(nlocal, dtype=pixels_dtype)
                    offset = global_offset
                    for istart, istop in period_ranges:
                        nn = istop - istart
                        dslice = slice(idet*nsamp + offset,
                                       idet*nsamp + offset + nn)
                        pixels[istart:istop] = self._madam_pixels[dslice]
                        offset += nn
                    npix = 12*nside**2
                    good = np.logical_and(pixels >= 0, pixels < npix)
                    if not self._pixels_nested:
                        pixels[good] = hp.nest2ring(nside, pixels[good])
                    pixels[np.logical_not(good)] = -1
                    cachename = "{}_{}".format(self._pixels, det)
                    tod.cache.put(cachename, pixels, replace=True)
                global_offset = offset
        self._madam_pixels = None
        self._cache.destroy('pixels')

        if not self._purge_weights and nnz == nnz_full:
            # restore the weights from the Madam buffers
            global_offset = 0
            for obs, period_ranges in zip(data.obs, obs_period_ranges):
                tod = obs['tod']
                nlocal = tod.local_samples[1]
                for idet, det in enumerate(detectors):
                    weights = np.zeros([nlocal, nnz], dtype=weight_dtype)
                    offset = global_offset
                    for istart, istop in period_ranges:
                        nn = istop - istart
                        dwslice = slice((idet*nsamp+offset) * nnz,
                                        (idet*nsamp+offset+nn) * nnz)
                        weights[istart:istop] \
                            = self._madam_pixweights[dwslice].reshape([-1, nnz])
                        offset += nn
                    cachename = "{}_{}".format(self._weights, det)
                    tod.cache.put(cachename, weights, replace=True)
                global_offset = offset
        self._madam_pixweights = None
        self._cache.destroy('pixweights')

        return weight_dtype
