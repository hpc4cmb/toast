# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import re
from typing import Type

import h5py
import numpy as np
from astropy import units as u

from .timing import Timer, function_timer
from .utils import Logger, hdf5_use_serial, name_UID


class Noise(object):
    """Noise objects act as containers for noise PSDs.

    Noise is a base class for an object that describes the noise properties of local
    detectors for a single observation.  The PSDs in the input dictionary should be
    arrays with units (Quantities).

    Args:
        detectors (list): Names of detectors.
        freqs (dict): Dictionary of arrays of frequencies for `psds`.
        psds (dict): Dictionary of array Quantities which contain the PSD values for
            each detector or `mixmatrix` key.
        mixmatrix (dict): Mixing matrix describing how the PSDs should be combined for
            each detector noise model.  If provided, must contain entries for every
            detector, and every key specified for a detector must be defined in `freqs`
            and `psds`.
        indices (dict):  Integer index for every PSD, useful for generating
            indepedendent and repeateable noise realizations. If absent, running
            indices will be assigned and provided.
        detweights (dict):  If not None, override internal logic used to determine
            detector noise weights.  If overridden, noise weights should reflect
            inverse white noise variance per sample (and have units).

    Attributes:
        detectors (list): List of detector names
        keys (list): List of PSD names

    Raises:
        KeyError: If `freqs`, `psds`, `mixmatrix` or `indices` do not include all
            relevant entries.
        ValueError: If vector lengths in `freqs` and `psds` do not match.

    """

    @function_timer
    def __init__(
        self,
        detectors=list(),
        freqs=dict(),
        psds=dict(),
        mixmatrix=None,
        indices=None,
        detweights=None,
    ):
        self._dets = list(sorted(detectors))
        if mixmatrix is None:
            # Default diagonal mixing matrix
            self._keys = self._dets
            self._keys_for_dets = {x: [x] for x in self._dets}
            self._dets_for_keys = {x: [x] for x in self._dets}
            self._mixmatrix = {d: {d: 1.0} for d in self._dets}
        else:
            # Assemble the list of keys needed for the specified detectors
            self._mixmatrix = dict()
            for det in self._dets:
                self._mixmatrix[det] = dict()
                for key, weight in mixmatrix[det].items():
                    self._mixmatrix[det][key] = weight
            self._init_lookup()

        if indices is None:
            self._indices = {x: name_UID(x) for x in self._keys}
        else:
            self._indices = dict(indices)
        self._freqs = {}
        self._psds = {}
        self._rates = {}

        for key in self._keys:
            if psds[key].shape[0] != freqs[key].shape[0]:
                raise ValueError("PSD length must match the number of frequencies")
            if not isinstance(freqs[key], u.Quantity):
                raise TypeError("Each frequency array must be a Quantity")
            if not isinstance(psds[key], u.Quantity):
                raise TypeError("Each PSD array must be a Quantity")
            # Ensure that the frequencies are convertible to expected units
            try:
                test_convert = freqs[key].to(u.Hz)
            except Exception:
                raise ValueError("Each frequency array must be convertible to Hz")
            # Ensure that the PSDs are convertible to expected units
            try:
                test_convert = psds[key].to(u.K**2 * u.second)
            except Exception:
                raise ValueError("Each PSD must be convertible to K**2 * s")
            self._freqs[key] = np.copy(freqs[key].to(u.Hz))
            self._psds[key] = np.copy(psds[key].to(u.K**2 * u.second))
            # last frequency point should be Nyquist
            self._rates[key] = 2.0 * self._freqs[key][-1]

        self._detweights = detweights

    def _init_lookup(self):
        # Initialize the detector / stream mapping structures
        keys = set()
        self._keys_for_dets = dict()
        self._dets_for_keys = dict()
        for det in self._dets:
            self._keys_for_dets[det] = list()
            for key, weight in self._mixmatrix[det].items():
                keys.add(key)
                if key not in self._dets_for_keys:
                    self._dets_for_keys[key] = list()
                if weight != 0:
                    self._keys_for_dets[det].append(key)
                    self._dets_for_keys[key].append(det)
        self._keys = list(sorted(keys))

    @property
    def detectors(self):
        """(list): list of strings containing the detector names."""
        return self._dets

    @property
    def keys(self):
        """(list): list of strings containing the PSD names."""
        return self._keys

    @property
    def mixing_matrix(self):
        """(dict): the full mixing matrix."""
        return self._mixmatrix

    def multiply_ntt(self, key, data):
        """Filter the data with noise covariance."""
        raise NotImplementedError("multiply_ntt not yet implemented")

    def multiply_invntt(self, key, data):
        """Filter the data with inverse noise covariance."""
        raise NotImplementedError("multiply_invntt not yet implemented")

    def weight(self, det, key):
        """Return the mixing weight for noise `key` in `det`.

        Args:
            det (str): Detector name
            key (std): Mixing matrix key.
        Returns:
            weight (float): Mixing matrix weight

        """
        weight = 0
        if key in self._mixmatrix[det]:
            weight = self._mixmatrix[det][key]
        return weight

    @function_timer
    def all_keys_for_dets(self, dets):
        """Return a complete list of noise keys that have nonzero
        weights for given detectors:
        """
        keys = set()
        for det in dets:
            keys.update(self._keys_for_dets[det])
        return list(sorted(keys))

    def index(self, key):
        """Return the PSD index for `key`

        Args:
            key (std): Detector name or mixing matrix key.
        Returns:
            index (int): PSD index.

        """
        return self._indices[key]

    def freq(self, key):
        """Get the frequencies corresponding to `key`.

        Args:
            key (str): Detector name or mixing matrix key.
        Returns:
            (array): Frequency bins that are used for the PSD.

        """
        return self._freqs[key]

    def rate(self, key):
        """Get the sample rate for `key`.

        Args:
            key (str): the detector name or mixing matrix key.
        Returns:
            (float): the sample rate in Hz.

        """
        return self._rates[key]

    def psd(self, key):
        """Get the PSD corresponding to `key`.

        Args:
            key (str): Detector name or mixing matrix key.
        Returns:
            (array): PSD matching the key.

        """
        return self._psds[key]

    def _detector_weight(self, det):
        """Internal function which can be overridden by derived classes."""
        if self._detweights is None:
            # Compute an effective scalar "noise weight" for each detector based on the
            # white noise level, accounting for the fact that the PSD may have a
            # transfer function roll-off near Nyquist
            self._detweights = {d: 0.0 * (1.0 / u.K**2) for d in self.detectors}
            mix = self.mixing_matrix
            for k in self.keys:
                freq = self.freq(k)
                psd = self.psd(k)
                rate = self.rate(k)
                # Noise in the middle of the PSD
                first = np.searchsorted(freq, rate * 0.225, side="left")
                last = np.searchsorted(freq, rate * 0.275, side="right")
                if first == last:
                    first = max(0, first - 1)
                    last = min(freq.size - 1, last + 1)
                noisevar_mid = np.median(psd[first:last])
                # Noise in the end of the PSD
                first = np.searchsorted(freq, rate * 0.45, side="left")
                last = np.searchsorted(freq, rate * 0.50, side="right")
                if first == last:
                    first = max(0, first - 1)
                    last = min(freq.size - 1, last + 1)
                noisevar_end = np.median(psd[first:last])
                if noisevar_end / noisevar_mid < 0.5:
                    # There is a transfer function roll-off.  Measure the
                    # white noise plateau value
                    first = np.searchsorted(freq, rate * 0.2, side="left")
                    last = np.searchsorted(freq, rate * 0.4, side="right")
                else:
                    # No significant roll-off.  Use the last PSD bins for
                    # the noise variance
                    first = np.searchsorted(freq, rate * 0.45, side="left")
                    last = np.searchsorted(freq, rate * 0.50, side="right")
                if first == last:
                    first = max(0, first - 1)
                    last = min(freq.size - 1, last + 1)
                noisevar = np.median(psd[first:last])
                invvar = (1.0 / noisevar / rate).decompose()
                for det in self._dets_for_keys[k]:
                    self._detweights[det] += mix[det][k] * invvar
        return self._detweights[det]

    def detector_weight(self, det):
        """Return the relative noise weight for a detector.

        Args:
            det (str):  The detector name.

        Returns:
            (float):  The noise weight for this detector.

        """
        return self._detector_weight(det)

    @function_timer
    def _gather_base(self, comm):
        out = None

        # Detector weights might be customized by derived classes.  Get the list
        # of weights locally before gathering.
        local_weights = {d: self.detector_weight(d) for d in self._dets}

        if comm is None or comm.size == 1:
            out = {
                "dets": self._dets,
                "freqs": self._freqs,
                "psds": self._psds,
                "mix": self._mixmatrix,
                "indices": self._indices,
                "weights": local_weights,
            }
            return out

        # First gather the stream properties and remove duplicates.  Each stream
        # index should be globally unique and streams with the same index on different
        # processes must have identical PSDs.

        all_indices = comm.gather(self._indices, root=0)
        all_freqs = comm.gather(self._freqs, root=0)
        all_psds = comm.gather(self._psds, root=0)

        if comm.rank == 0:
            out = dict()
            out["indices"] = dict()
            out["freqs"] = dict()
            out["psds"] = dict()
            for pind, pfreq, ppsd in zip(all_indices, all_freqs, all_psds):
                for k, v in pind.items():
                    if k not in out["indices"]:
                        out["indices"][k] = v
                        out["freqs"][k] = pfreq[k]
                        out["psds"][k] = ppsd[k]

        del all_indices
        del all_freqs
        del all_psds

        # Build full detector list and mixing matrix

        all_dets = comm.gather(self._dets, root=0)
        all_mix = comm.gather(self._mixmatrix, root=0)

        if comm.rank == 0:
            gdets = set()
            out["mix"] = dict()
            for pdets, pmix in zip(all_dets, all_mix):
                for det in pdets:
                    if det in gdets:
                        msg = f"Multiple processes have detector '{det}'.  "
                        msg += f"Are you using the grid column communicator?"
                        raise RuntimeError(msg)
                    gdets.add(det)
                    out["mix"][det] = dict()
                    for k, v in pmix[det].items():
                        out["mix"][det][k] = v
            out["dets"] = list(sorted(gdets))

        del all_dets
        del all_mix

        # Detector weights

        all_weights = comm.gather(local_weights, root=0)

        if comm.rank == 0:
            out["weights"] = dict()
            for pweight in all_weights:
                for k, v in pweight.items():
                    out["weights"][k] = v

        del all_weights
        return out

    def _gather(self, comm):
        return self._gather_base(comm)

    @function_timer
    def gather(self, comm):
        """Gather local pieces of the noise model to one process.

        Every process has the noise model for its local detectors.  For doing I/O,
        we need to combine these pieces into a single model on one process.  The
        MPI communicator would typically be just the first process grid column.

        Args:
            comm (MPI.Comm):  The communicator used to gather the pieces.

        Returns:
            (dict):  The properties of the full noise model on the rank zero
                process and None on other processes.

        """
        return self._gather(comm)

    @function_timer
    def _scatter_base(
        self,
        comm,
        local_dets,
        props,
    ):
        if comm is None or comm.size == 1:
            self._dets = local_dets
            self._freqs = props["freqs"]
            self._psds = props["psds"]
            self._indices = props["indices"]
            self._mixmatrix = props["mix"]
            self._detweights = props["weights"]
            self._rates = {d: 2.0 * x[-1] for d, x in props["freqs"].items()}
        else:
            # Local detectors
            self._dets = local_dets

            # Get local detector weights
            if comm.rank == 0:
                all_weights = comm.bcast(props["weights"], root=0)
            else:
                all_weights = comm.bcast(None, root=0)
            self._detweights = dict()
            for d in self._dets:
                self._detweights[d] = all_weights[d]
            del all_weights

            # Local mixing matrix
            if comm.rank == 0:
                all_mix = comm.bcast(props["mix"], root=0)
            else:
                all_mix = comm.bcast(None, root=0)
            local_streams = set()
            self._mixmatrix = dict()
            for d in self._dets:
                self._mixmatrix[d] = dict()
                for k, v in all_mix[d].items():
                    local_streams.add(k)
                    self._mixmatrix[d][k] = v
            del all_mix

            # Local streams
            if comm.rank == 0:
                all_freqs = comm.bcast(props["freqs"], root=0)
                all_psds = comm.bcast(props["psds"], root=0)
                all_indices = comm.bcast(props["indices"], root=0)
            else:
                all_freqs = comm.bcast(None, root=0)
                all_psds = comm.bcast(None, root=0)
                all_indices = comm.bcast(None, root=0)
            self._freqs = dict()
            self._psds = dict()
            self._indices = dict()
            self._rates = dict()
            for ls in local_streams:
                self._freqs[ls] = all_freqs[ls]
                self._psds[ls] = all_psds[ls]
                self._indices[ls] = all_indices[ls]
                self._rates[ls] = 2.0 * self._freqs[ls][-1]
            del all_freqs
            del all_psds
            del all_indices
        self._init_lookup()

    def _scatter(self, comm, local_dets, props):
        self._scatter_base(comm, local_dets, props)

    @function_timer
    def scatter(
        self,
        comm,
        local_dets,
        props,
    ):
        """Distribute noise model properties.

        Given the global set of properties for all detectors, distribute this
        so that every process has only the information needed for its local
        detectors.  The internal data is replaced.

        Args:
            comm (MPI.Comm):  The communicator over which to distribute the pieces.
            local_dets (list):  The list of local detectors on this process.
            props (dict):  The dictionary of all properties for all streams.

        Returns:
            None

        """
        self._scatter(comm, local_dets, props)

    def _redistribute(self, old_dist, new_dist):
        # Every column of the process grid has the same information.  We
        # gather the full noise model to rank zero.
        props = None
        if old_dist.comm_row_rank == 0:
            props = self._gather(old_dist.comm_col)

        # Broadcast this full noise model along the first process row of the
        # new distribution.
        if new_dist.comm_col_rank == 0 and new_dist.comm_row_size > 1:
            props = new_dist.comm_row.bcast(props, root=0)

        # Scatter noise model down each process column.
        self._scatter(
            new_dist.comm_col,
            new_dist.dets[new_dist.comm.group_rank],
            props,
        )

    def redistribute(self, old_dist, new_dist):
        """Redistribute noise model.

        Args:
            old_dist (DistDetSamp):  The current distribution.
            new_dist (DistDetSamp):  The new distribution.

        Returns:
            None

        """
        self._redistribute(old_dist, new_dist)

    @function_timer
    def _save_base_hdf5(self, hf, obs):
        """Write internal data to an open HDF5 group.

        This is collective, since each process has a local subset of detectors.
        We communicate the contents to one process which writes them to the
        open HDF5 group.

        Args:
            hf (Group):  The HDF5 group in which to write the datasets
            obs (Observation):  The parent observation

        Returns:
            None

        """
        gcomm = obs.comm.comm_group
        rank = 0
        nproc = 1
        if gcomm is not None:
            rank = gcomm.rank
            nproc = gcomm.size

        # Gather data to the rank zero of the first grid column
        props = None
        if obs.comm_row_rank == 0:
            props = self._gather_base(obs.comm_col)

        # First store the mixing matrix as a separate dataset, and find the maximum
        # string length used for the keys.  Also write the detector weights.

        n_mix = 0
        n_det = 0
        maxstr = 0
        mixdtype = None
        wtdtype = None
        wtunit = None
        if rank == 0:
            mixdata = list()
            maxstr = 0
            for det, streams in props["mix"].items():
                maxstr = max(maxstr, len(det))
                for strm, weight in streams.items():
                    maxstr = max(maxstr, len(strm))
                    mixdata.append((det, strm, weight))
            maxstr += 1
            mixdtype = np.dtype(f"a{maxstr}, a{maxstr}, f4")
            wtdtype = np.dtype(f"a{maxstr}, f4")
            n_mix = len(mixdata)
            n_det = len(props["dets"])
            wtunit = props["weights"][props["dets"][0]].unit

        if gcomm is not None:
            mixdtype = gcomm.bcast(mixdtype, root=0)
            wtdtype = gcomm.bcast(wtdtype, root=0)
            n_mix = gcomm.bcast(n_mix, root=0)
            n_det = gcomm.bcast(n_det, root=0)
            wtunit = gcomm.bcast(wtunit, root=0)
            maxstr = gcomm.bcast(maxstr, root=0)

        wds = None
        mds = None

        if hf is not None:
            # This process is participating
            wds = hf.create_dataset("detector_weights", (n_det,), dtype=wtdtype)
            wds.attrs["unit"] = str(wtunit)
            mds = hf.create_dataset("mixing_matrix", (n_mix,), dtype=mixdtype)

        # Rank zero does the writing, and is guaranteed to always be
        # participating in the write.
        if rank == 0:
            wtdata = list()
            for d in props["dets"]:
                wtdata.append((d, props["weights"][d].value))
            packed = np.array(wtdata, dtype=wtdtype)
            wds.write_direct(packed)
            packed = np.array(mixdata, dtype=mixdtype)
            mds.write_direct(packed)
            del packed

        if gcomm is not None:
            gcomm.barrier()
        del wds
        del mds

        # Each stream psd can potentially have a unique set of frequencies, but in
        # practice we usually have a just one or a few.  To reduce the number of
        # datasets created, we group PSDs according to the frequency vectors.
        #
        # NOTE:  We assume here that any frequency array whose first and last several
        # elements are equal are themselves equal.  This should cover all cases.
        # Also note that small bit differences in floating point frequency values can
        # result in different hash values used in the dataset names.  Because of this,
        # we compute the grouping of PSDs on one process and broadcast.

        psd_group = dict()
        if rank == 0:
            # Compute the frequency hash on one process
            for k, v in props["freqs"].items():
                fhash = (
                    hash(v[2] * 1000 + v[-2] + v[-1])
                    .to_bytes(8, "big", signed=True)
                    .hex()
                )
                psd_group[k] = fhash
        if gcomm is not None:
            psd_group = gcomm.bcast(psd_group)

        # Organize the PSD information in groups according to the frequency arrays.
        # Also verify that all PSD units match.
        psd_set_meta = dict()
        if rank == 0:
            psd_sets = dict()
            for k, v in props["freqs"].items():
                fhash = psd_group[k]
                if fhash not in psd_sets:
                    psd_sets[fhash] = {
                        "freq": v,
                        "indices": list(),
                        "psds": list(),
                        "keys": list(),
                    }
                    psd_set_meta[fhash] = {
                        "n_freq": len(v),
                        "n_psd": 0,
                        "units": None,
                    }
                if psd_set_meta[fhash]["units"] is None:
                    psd_set_meta[fhash]["units"] = props["psds"][k].unit
                else:
                    if psd_set_meta[fhash]["units"] != props["psds"][k].unit:
                        raise RuntimeError(
                            "All PSD units in a Noise object must be the same"
                        )
                psd_set_meta[fhash]["n_psd"] += 1
                psd_sets[fhash]["psds"].append(props["psds"][k])
                psd_sets[fhash]["indices"].append(props["indices"][k])
                psd_sets[fhash]["keys"].append(k)
        if gcomm is not None:
            psd_set_meta = gcomm.bcast(psd_set_meta, root=0)

        # Add an attribute for the units
        if hf is not None:
            first_hash = list(psd_set_meta.keys())[0]
            ustr = str(psd_set_meta[first_hash]["units"])
            hf.attrs["psd_units"] = ustr

        # Create a dataset for each set of PSDs.  Also create separate datasets
        # for the name and index of each PSD.
        for fhash, meta in psd_set_meta.items():
            if rank == 0:
                setprops = psd_sets[fhash]
                freq = setprops["freq"]
                psds = setprops["psds"]
                indx = setprops["indices"]
                keys = setprops["keys"]
            nrows = 1 + meta["n_psd"]
            ncols = meta["n_freq"]

            pds = None
            inds = None
            kds = None
            if hf is not None:
                # This process is participating
                pds = hf.create_dataset(fhash, (nrows, ncols), dtype=np.float32)
                inds = hf.create_dataset(
                    f"{fhash}_indices", (meta["n_psd"],), dtype=np.uint32
                )
                keytype = np.dtype(f"a{maxstr}")
                kds = hf.create_dataset(
                    f"{fhash}_keys", (meta["n_psd"],), dtype=keytype
                )

            # Rank zero does the writing, and is guaranteed to always be
            # participating in the write.
            if rank == 0:
                packed = np.zeros((nrows, ncols), dtype=np.float32)
                packed[0] = freq
                packed[1:] = np.array(psds, dtype=np.float32)
                pds.write_direct(packed)

                packed = np.array(indx, dtype=np.uint32)
                inds.write_direct(packed)

                packed = np.array(keys, dtype=keytype)
                kds.write_direct(packed)
                del packed

            # Wait for writing to complete
            if gcomm is not None:
                gcomm.barrier()

            # Close datasets
            del pds
            del inds
            del kds

    def _save_hdf5(self, handle, obs, **kwargs):
        """Internal method which can be overridden by derived classes."""
        self._save_base_hdf5(handle, obs)

    def save_hdf5(self, handle, obs, **kwargs):
        """Save the noise object to an HDF5 file.

        Args:
            handle (h5py.Group):  The group to populate.
            obs (Observation):  The parent observation.

        Returns:
            None

        """
        gcomm = obs.comm.comm_group
        if (gcomm is None) or (gcomm.rank == 0):
            # The rank zero process should always be writing
            if handle is None:
                raise RuntimeError("HDF5 group is not open on the root process")
        _ = self.detector_weight(self.detectors[0])
        self._save_hdf5(handle, obs, **kwargs)

    @function_timer
    def _load_base_hdf5(self, hf, obs):
        """Read internal data from an open HDF5 group"""
        gcomm = obs.comm.comm_group
        rank = 0
        nproc = 1
        if gcomm is not None:
            rank = gcomm.rank
            nproc = gcomm.size

        props = None
        if rank == 0:
            props = dict()
        psd_units = None

        indx_pat = re.compile(r"(.*)_indices")
        key_pat = re.compile(r"(.*)_keys")

        if hf is not None:
            # This process is participating in dataset open.  However, only one
            # process will load data from the datasets.  The result will be
            # scattered at the end.

            # Load the weights
            dset = hf["detector_weights"]
            wtunit = u.Unit(dset.attrs["unit"])
            if rank == 0:
                props["dets"] = list()
                props["weights"] = dict()
                for det, val in dset:
                    det = det.decode("utf-8")
                    props["dets"].append(det)
                    props["weights"][det] = u.Quantity(val, wtunit)

            # Now the mixing matrix
            dset = hf["mixing_matrix"]
            if rank == 0:
                props["freqs"] = dict()
                props["psds"] = dict()
                props["indices"] = dict()
                props["mix"] = dict()
                for det, key, val in hf["mixing_matrix"]:
                    det = det.decode("utf-8")
                    key = key.decode("utf-8")
                    if det not in props["mix"]:
                        props["mix"][det] = dict()
                    props["mix"][det][key] = val

            # Get the units
            psd_units = u.Unit(hf.attrs["psd_units"])

            for dsname in hf.keys():
                if indx_pat.match(dsname) is not None:
                    # Will be processed as part of the associated dataset below
                    continue
                if key_pat.match(dsname) is not None:
                    # Will be processed as part of the associated dataset below
                    continue
                if dsname == "mixing_matrix":
                    # Already processed above
                    continue
                if dsname == "detector_weights":
                    # Already processed above
                    continue
                indx_name = f"{dsname}_indices"
                keys_name = f"{dsname}_keys"
                if indx_name not in hf.keys() and keys_name not in hf.keys():
                    # This is not a PSD set
                    continue

                # Before loading the PSD data, load the stream names and indices
                kds = hf[keys_name]
                if rank == 0:
                    psd_keys = [x.decode() for x in kds]
                del kds

                ids = hf[indx_name]
                if rank == 0:
                    psd_indices = np.array(ids, dtype=np.uint32)
                del ids

                pds = hf[dsname]
                if rank == 0:
                    freq = pds[0]
                    for key, indx, psdrow in zip(psd_keys, psd_indices, pds[1:]):
                        props["indices"][key] = int(indx)
                        props["freqs"][key] = u.Quantity(freq, u.Hz)
                        props["psds"][key] = u.Quantity(psdrow, psd_units)
                del pds

        # The data now exists on the rank zero process of the group.  If we have
        # multiple processes along each row, broadcast data to the other processes
        # in the first row.
        if obs.comm_row_size > 1 and obs.comm_col_rank == 0:
            props = obs.comm_row.bcast(props, root=0)

        # Scatter across each column of the process grid
        self._scatter_base(
            obs.comm_col,
            obs.local_detectors,
            props,
        )

    def _load_hdf5(self, handle, obs, **kwargs):
        """Internal method which can be overridden by derived classes."""
        self._load_base_hdf5(handle, obs)

    def load_hdf5(self, handle, obs, **kwargs):
        """Load the noise object from an HDF5 file.

        Args:
            handle (h5py.Group):  The group containing noise model.
            obs (Observation):  The parent observation.

        Returns:
            None

        """
        gcomm = obs.comm.comm_group
        if (gcomm is None) or (gcomm.rank == 0):
            # The rank zero process should always be reading
            if handle is None:
                raise RuntimeError("HDF5 group is not open on the root process")
        self._load_hdf5(handle, obs, **kwargs)

    def __repr__(self):
        mix_min = np.min([len(y) for x, y in self._mixmatrix.items()])
        mix_max = np.max([len(y) for x, y in self._mixmatrix.items()])
        value = f"<Noise model with {len(self._dets)} detectors each built from "
        value += f"between {mix_min} and {mix_max} independent streams"
        value += ">"
        return value

    def __eq__(self, other):
        log = Logger.get()
        fail = 0
        if set(self._dets) != set(other._dets):
            log.verbose(f"Noise __eq__:  dets {set(self._dets)} != {set(other._dets)}")
            fail = 1
        elif set(self._keys) != set(other._keys):
            log.verbose(f"Noise __eq__:  keys {set(self._keys)} != {set(other._keys)}")
            fail = 1
        elif self._rates != other._rates:
            log.verbose(f"Noise __eq__:  rates {self._rates} != {other._rates}")
            fail = 1
        elif self._indices != other._indices:
            log.verbose(f"Noise __eq__:  indices {self._indices} != {other._indices}")
            fail = 1
        elif self._mixmatrix != other._mixmatrix:
            log.verbose(f"Noise __eq__:  mix {self._mixmatrix} != {other._mixmatrix}")
            fail = 1
        else:
            for k, v in self._freqs.items():
                if not np.allclose(v.to_value(u.Hz), other._freqs[k].to_value(u.Hz)):
                    log.verbose(f"Noise __eq__:  freqs[{k}] {v} != {other._freqs[k]}")
                    fail = 1
            for k, v in self._psds.items():
                if not np.allclose(
                    v.to_value(u.K**2 * u.second),
                    other._psds[k].to_value(u.K**2 * u.second),
                ):
                    log.verbose(f"Noise __eq__:  psds[{k}] {v} != {other._psds[k]}")
                    fail = 1
        return fail == 0

    def __ne__(self, other):
        return not self.__eq__(other)
