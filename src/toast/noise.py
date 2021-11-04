# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from typing import Type
import re
import numpy as np

import hashlib

from astropy import units as u

import h5py

from .timing import function_timer

from .utils import have_hdf5_parallel


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
            self._mixmatrix = dict()
            for d in self._dets:
                self._mixmatrix[d] = {d: 1.0}
        else:
            # Assemble the list of keys needed for the specified detectors
            keys = set()
            self._mixmatrix = {}
            for det in self._dets:
                self._mixmatrix[det] = {}
                for key, weight in mixmatrix[det].items():
                    keys.add(key)
                    self._mixmatrix[det][key] = weight
            self._keys = list(sorted(keys))
        if indices is None:
            self._indices = {}
            for i, key in enumerate(self._keys):
                self._indices[key] = i
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
                test_convert = psds[key].to_value(u.K ** 2 * u.second)
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
        keys = []
        for det in dets:
            for key, value in self._mixmatrix[det].items():
                if value != 0:
                    keys.append(key)
        return keys

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
            for k in self.keys:
                freq = self.freq(k)
                psd = self.psd(k)
                rate = self.rate(k)
                ind = np.logical_and(freq > rate * 0.2, freq < rate * 0.4)
                noisevar = np.median(psd[ind].to_value(u.K ** 2 * u.second))
                for det in self.detectors:
                    wt = self.weight(det, k)
                    if wt != 0.0:
                        self._detweights[det] += wt * (1.0 / noisevar)
        return self._detweights[det]

    def detector_weight(self, det):
        """Return the relative noise weight for a detector.

        Args:
            det (str):  The detector name.

        Returns:
            (float):  The noise weight for this detector.

        """
        return self._detector_weight(det)

    def _save_base_hdf5(self, hf, comm=None, force_serial=False):
        """Write internal data to an open HDF5 group."""
        parallel = have_hdf5_parallel()
        if force_serial:
            parallel = False
        participating = parallel or comm is None or comm.rank == 0

        # First store the mixing matrix as a separate dataset, and find the maximum
        # string length used for the keys.

        mixdata = list()
        maxstr = 0
        for det, streams in self._mixmatrix.items():
            maxstr = max(maxstr, len(det))
            for strm, weight in streams.items():
                maxstr = max(maxstr, len(strm))
                mixdata.append((det, strm, weight))
        maxstr += 1
        mixdtype = np.dtype(f"a{maxstr}, a{maxstr}, f8")
        if participating:
            ds = hf.create_dataset("mixing_matrix", (len(mixdata),), dtype=mixdtype)
            if comm is None or comm.rank == 0:
                ds[:] = np.array(mixdata, dtype=mixdtype)
            del ds

        # Each stream psd can potentially have a unique set of frequencies, but in
        # practice we usually have a just one or a few.  To reduce the number of
        # datasets created, we group PSDs according to the frequency vectors.
        #
        # NOTE:  We assume here that any frequency array whose first and last several
        # elements are equal are themselves equal.  This should cover all cases.

        psd_sets = dict()
        for k in self.keys:
            freq = self.freq(k)
            freq_str = f"{freq[:2]} {freq[-3:]}"
            fhash = hashlib.md5(freq_str.encode("utf8")).hexdigest()
            if fhash not in psd_sets:
                psd_sets[fhash] = {
                    "freq": freq,
                    "indices": list(),
                    "psds": list(),
                    "keys": list(),
                }
            psd_sets[fhash]["psds"].append(self.psd(k))
            psd_sets[fhash]["indices"].append(self.index(k))
            psd_sets[fhash]["keys"].append(k)

        # Create a dataset for each set of PSDs.  Also create separate datasets
        # for the name and index of each PSD.
        if participating:
            for fhash, props in psd_sets.items():
                freq = props["freq"]
                psds = props["psds"]
                indx = props["indices"]
                keys = props["keys"]
                nrows = 1 + len(psds)
                ncols = len(freq)
                ds = hf.create_dataset(fhash, (nrows, ncols), dtype=np.float64)
                if comm is None or comm.rank == 0:
                    ds[0] = freq
                    ds[1:] = np.array(psds, dtype=np.float64)
                del ds
                ds = hf.create_dataset(f"{fhash}_indices", len(psds), dtype=np.int32)
                if comm is None or comm.rank == 0:
                    ds[:] = indx
                del ds
                keytype = np.dtype(f"a{maxstr}")
                ds = hf.create_dataset(f"{fhash}_keys", len(psds), dtype=keytype)
                if comm is None or comm.rank == 0:
                    ds[:] = np.array(keys, dtype=keytype)
                del ds

    def _save_hdf5(self, handle, comm=None, force_serial=False, **kwargs):
        """Internal method which can be overridden by derived classes."""
        parallel = have_hdf5_parallel()
        if force_serial:
            parallel = False
        if isinstance(handle, h5py.Group):
            self._save_base_hdf5(handle, comm=comm, force_serial=force_serial)
        else:
            hf = None
            if parallel:
                hf = h5py.File(handle, "w", driver="mpio", comm=comm)
            elif comm is None or comm.rank == 0:
                hf = h5py.File(handle, "w")
            self._save_base_hdf5(hf, comm=comm, force_serial=force_serial)
            if hf is not None:
                hf.flush()
                hf.close()
            del hf

    def save_hdf5(self, handle, comm=None, force_serial=False, **kwargs):
        """Save the noise object to an HDF5 file.

        Args:
            handle (str, file object, group):  Any object accepted by the h5py
                package.

        Returns:
            None

        """
        self._save_hdf5(handle, comm=comm, force_serial=force_serial, **kwargs)

    def _load_base_hdf5(self, hf, comm=None):
        """Read internal data from an open HDF5 group"""
        self._freqs = dict()
        self._psds = dict()
        self._rates = dict()
        self._indices = dict()
        self._mixmatrix = dict()
        indx_pat = re.compile(r"(.*)_indices")
        key_pat = re.compile(r"(.*)_keys")

        # First load the mixing matrix, which provides all the key and detector names

        dets = set()
        keys = set()
        for det, key, val in hf["mixing_matrix"]:
            det = det.decode("utf-8")
            key = key.decode("utf-8")
            dets.add(det)
            keys.add(key)
            if det not in self._mixmatrix:
                self._mixmatrix[det] = dict()
            self._mixmatrix[det][key] = val
        self._keys = list(sorted(keys))
        self._dets = list(sorted(dets))

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
            ds = hf[keys_name]
            psd_keys = [x.decode() for x in ds]
            del ds
            ds = hf[indx_name]
            psd_indices = list(ds)
            del ds

            ds = hf[dsname]
            freq = ds[0]
            rate = 2.0 * freq[-1]
            for key, indx, psdrow in zip(psd_keys, psd_indices, ds[1:]):
                self._rates[key] = rate * u.Hz
                self._indices[key] = indx
                self._freqs[key] = u.Quantity(freq, u.Hz)
                self._psds[key] = u.Quantity(psdrow, u.K ** 2 * u.second)
            del ds

    def _load_hdf5(self, handle, comm=None, **kwargs):
        """Internal method which can be overridden by derived classes."""
        if isinstance(handle, h5py.Group):
            self._load_base_hdf5(handle, comm=comm)
        else:
            with h5py.File(handle, "r") as hf:
                self._load_base_hdf5(hf, comm=comm)

    def load_hdf5(self, handle, comm=None, **kwargs):
        """Load the noise object from an HDF5 file.

        Args:
            handle (str, file object, group):  Any object accepted by the h5py
                package.

        Returns:
            None

        """
        self._load_hdf5(handle, comm=comm, **kwargs)

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
            if not np.allclose(v, other._freqs[k]):
                return False
        for k, v in self._psds.items():
            if not np.allclose(v, other._psds[k]):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)
