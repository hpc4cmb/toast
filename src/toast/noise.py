# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from typing import Type
import numpy as np

from astropy import units as u


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

    def __init__(self, detectors, freqs, psds, mixmatrix=None, indices=None):
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
        weight = 0.0
        if key in self._mixmatrix[det]:
            weight = self._mixmatrix[det][key]
        return weight

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
                noisevar = np.median(psd[ind].to_value(u.K**2 * u.second))
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
