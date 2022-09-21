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
from .utils import hdf5_use_serial


class Noise(object):
    """Noise objects act as containers for noise PSDs.

    Noise is a base class for an object that describes the noise properties of all
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
        self, detectors=list(), freqs=dict(), psds=dict(), mixmatrix=None, indices=None
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
            keys = set()
            self._mixmatrix = dict()
            self._keys_for_dets = dict()
            self._dets_for_keys = dict()
            for det in self._dets:
                self._keys_for_dets[det] = list()
                self._mixmatrix[det] = dict()
                for key, weight in mixmatrix[det].items():
                    keys.add(key)
                    if key not in self._dets_for_keys:
                        self._dets_for_keys[key] = list()
                    self._mixmatrix[det][key] = weight
                    if weight != 0:
                        self._keys_for_dets[det].append(key)
                        self._dets_for_keys[key].append(det)
            self._keys = list(sorted(keys))

        if indices is None:
            self._indices = {x: i for i, x in enumerate(self._keys)}
        else:
            self._indices = dict(indices)
        self._freqs = {}
        self._psds = {}
        self._rates = {}

        for key in self._keys:
            if psds[key].shape[0] != freqs[key].shape[0]:
                raise ValueError("PSD length must match the number of frequencies")
            if not isinstance(psds[key], u.Quantity):
                raise TypeError("Each PSD array must be a Quantity")
            # Ensure that the PSDs are convertible to expected units
            try:
                test_convert = psds[key].to_value(u.K**2 * u.second)
            except Exception:
                raise ValueError("Each PSD must be convertible to K**2 * s")
            self._freqs[key] = np.copy(freqs[key])
            self._psds[key] = np.copy(psds[key])
            # last frequency point should be Nyquist
            self._rates[key] = 2.0 * self._freqs[key][-1]

        self._detweights = None

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
            self._detweights = {d: 0.0 for d in self.detectors}
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
                noisevar_mid = np.median(psd[first:last].to_value(u.K**2 * u.second))
                # Noise in the end of the PSD
                first = np.searchsorted(freq, rate * 0.45, side="left")
                last = np.searchsorted(freq, rate * 0.50, side="right")
                if first == last:
                    first = max(0, first - 1)
                    last = min(freq.size - 1, last + 1)
                noisevar_end = np.median(psd[first:last].to_value(u.K**2 * u.second))
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
                noisevar = np.median(psd[first:last].to_value(u.K**2 * u.second))
                invvar = 1.0 / noisevar / rate.to_value(u.Hz)
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
    def _save_base_hdf5(self, hf, comm):
        """Write internal data to an open HDF5 group."""
        # First store the mixing matrix as a separate dataset, and find the maximum
        # string length used for the keys.

        rank = 0
        nproc = 1
        if comm is not None:
            rank = comm.rank
            nproc = comm.size

        mixdata = list()
        maxstr = 0
        for det, streams in self._mixmatrix.items():
            maxstr = max(maxstr, len(det))
            for strm, weight in streams.items():
                maxstr = max(maxstr, len(strm))
                mixdata.append((det, strm, weight))
        maxstr += 1
        mixdtype = np.dtype(f"a{maxstr}, a{maxstr}, f4")

        if hf is not None:
            # This process is participating
            mds = hf.create_dataset("mixing_matrix", (len(mixdata),), dtype=mixdtype)
            if rank == 0:
                # Only one process needs to write
                packed = np.array(mixdata, dtype=mixdtype)
                mds.write_direct(packed)
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
            for k in self.keys:
                freq = self.freq(k)
                fhash = (
                    hash(freq[2] * 1000 + freq[-2] + freq[-1])
                    .to_bytes(8, "big", signed=True)
                    .hex()
                )
                psd_group[k] = fhash
        if comm is not None:
            psd_group = comm.bcast(psd_group)

        # Organize the PSD information in groups according to the frequency arrays.
        # Also verify that all PSD units match.
        psd_sets = dict()
        psd_units = None
        for k in self.keys:
            freq = self.freq(k)
            fhash = psd_group[k]
            if fhash not in psd_sets:
                psd_sets[fhash] = {
                    "freq": freq,
                    "indices": list(),
                    "psds": list(),
                    "keys": list(),
                }
            if psd_units is None:
                psd_units = self.psd(k).unit
            else:
                if psd_units != self.psd(k).unit:
                    raise RuntimeError(
                        "All PSD units in a Noise object must be the same"
                    )
            psd_sets[fhash]["psds"].append(self.psd(k))
            psd_sets[fhash]["indices"].append(self.index(k))
            psd_sets[fhash]["keys"].append(k)

        # Add an attribute for the units
        if hf is not None:
            hf.attrs["psd_units"] = str(psd_units)

        # Create a dataset for each set of PSDs.  Also create separate datasets
        # for the name and index of each PSD.
        for fhash, props in psd_sets.items():
            freq = props["freq"]
            psds = props["psds"]
            indx = props["indices"]
            keys = props["keys"]
            nrows = 1 + len(psds)
            ncols = len(freq)
            if hf is not None:
                # This process is participating
                pds = hf.create_dataset(fhash, (nrows, ncols), dtype=np.float32)
                if rank == 0:
                    packed = np.zeros((nrows, ncols), dtype=np.float32)
                    packed[0] = freq
                    packed[1:] = np.array(psds, dtype=np.float32)
                    pds.write_direct(packed)
                del pds
                ids = hf.create_dataset(
                    f"{fhash}_indices", (len(psds),), dtype=np.int32
                )
                if rank == 0:
                    packed = np.array(indx, dtype=np.int32)
                    ids.write_direct(packed)
                del ids
                keytype = np.dtype(f"a{maxstr}")
                kds = hf.create_dataset(f"{fhash}_keys", (len(psds),), dtype=keytype)
                if rank == 0:
                    packed = np.array(keys, dtype=keytype)
                    kds.write_direct(packed)
                del kds

    def _save_hdf5(self, handle, comm, **kwargs):
        """Internal method which can be overridden by derived classes."""
        self._save_base_hdf5(handle, comm)

    def save_hdf5(self, handle, comm=None, **kwargs):
        """Save the noise object to an HDF5 file.

        Args:
            handle (h5py.Group):  The group to populate.

        Returns:
            None

        """
        if (comm is None) or (comm.rank == 0):
            # The rank zero process should always be writing
            if handle is None:
                raise RuntimeError("HDF5 group is not open on the root process")
        self._save_hdf5(handle, comm, **kwargs)

    @function_timer
    def _load_base_hdf5(self, hf, comm):
        """Read internal data from an open HDF5 group"""

        rank = 0
        nproc = 1
        if comm is not None:
            rank = comm.rank
            nproc = comm.size

        self._freqs = dict()
        self._psds = dict()
        self._rates = dict()
        self._indices = dict()
        self._mixmatrix = dict()
        indx_pat = re.compile(r"(.*)_indices")
        key_pat = re.compile(r"(.*)_keys")

        # Determine if we need to broadcast results.  This occurs if only one process
        # has the file open but the communicator has more than one process.
        need_bcast = hdf5_use_serial(hf, comm)

        if hf is not None:
            # This process is participating.
            # First load the mixing matrix, which provides all the key and
            # detector names
            dets = set()
            keys = set()
            self._keys_for_dets = dict()
            self._dets_for_keys = dict()
            for det, key, val in hf["mixing_matrix"]:
                det = det.decode("utf-8")
                key = key.decode("utf-8")
                dets.add(det)
                keys.add(key)
                if det not in self._keys_for_dets:
                    self._keys_for_dets[det] = list()
                if key not in self._dets_for_keys:
                    self._dets_for_keys[key] = list()
                if det not in self._mixmatrix:
                    self._mixmatrix[det] = dict()
                self._mixmatrix[det][key] = val
                self._keys_for_dets[det].append(key)
                self._dets_for_keys[key].append(det)
            self._keys = list(sorted(keys))
            self._dets = list(sorted(dets))

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
                indx_name = f"{dsname}_indices"
                keys_name = f"{dsname}_keys"
                if indx_name not in hf.keys() and keys_name not in hf.keys():
                    # This is not a PSD set
                    continue
                # Before loading the PSD data, load the stream names and indices
                kds = hf[keys_name]
                psd_keys = [x.decode() for x in kds]
                del kds
                ids = hf[indx_name]
                psd_indices = list(ids)
                del ids

                pds = hf[dsname]
                freq = pds[0]
                rate = 2.0 * freq[-1]
                for key, indx, psdrow in zip(psd_keys, psd_indices, pds[1:]):
                    self._rates[key] = rate * u.Hz
                    self._indices[key] = indx
                    self._freqs[key] = u.Quantity(freq, u.Hz)
                    self._psds[key] = u.Quantity(psdrow, psd_units)
                del pds

        # Broadcast the results
        if need_bcast and comm is not None:
            self._keys = comm.bcast(self._keys, root=0)
            self._dets = comm.bcast(self._dets, root=0)
            self._rates = comm.bcast(self._rates, root=0)
            self._freqs = comm.bcast(self._freqs, root=0)
            self._psds = comm.bcast(self._psds, root=0)
            self._indices = comm.bcast(self._indices, root=0)
            self._mixmatrix = comm.bcast(self._mixmatrix, root=0)
            self._keys_for_dets = comm.bcast(self._keys_for_dets, root=0)
            self._dets_for_keys = comm.bcast(self._dets_for_keys, root=0)

    def _load_hdf5(self, handle, comm, **kwargs):
        """Internal method which can be overridden by derived classes."""
        self._load_base_hdf5(handle, comm)

    def load_hdf5(self, handle, comm=None, **kwargs):
        """Load the noise object from an HDF5 file.

        Args:
            handle (h5py.Group):  The group containing noise model.

        Returns:
            None

        """
        if (comm is None) or (comm.rank == 0):
            # The rank zero process should always be reading
            if handle is None:
                raise RuntimeError("HDF5 group is not open on the root process")
        self._load_hdf5(handle, comm, **kwargs)

    def __repr__(self):
        mix_min = np.min([len(y) for x, y in self._mixmatrix.items()])
        mix_max = np.max([len(y) for x, y in self._mixmatrix.items()])
        value = f"<Noise model with {len(self._dets)} detectors each built from "
        value += f"between {mix_min} and {mix_max} independent streams"
        value += ">"
        return value

    def __eq__(self, other):
        if self._dets != other._dets:
            return False
        if self._keys != other._keys:
            return False
        if self._rates != other._rates:
            return False
        if self._indices != other._indices:
            return False
        if self._mixmatrix != other._mixmatrix:
            return False
        for k, v in self._freqs.items():
            if not np.allclose(v.to_value(u.Hz), other._freqs[k].to_value(u.Hz)):
                return False
        for k, v in self._psds.items():
            if not np.allclose(
                v.to_value(u.K**2 * u.second),
                other._psds[k].to_value(u.K**2 * u.second),
            ):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)
