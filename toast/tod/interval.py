# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np


class Interval(object):
    """
    Class storing a time and sample range.
    """
    def __init__(self, start=None, stop=None, first=None, last=None):
        self._start = start
        self._stop = stop
        self._first = first
        self._last = last

    @property
    def start(self):
        if self._start is None:
            raise RuntimeError("Start time is not yet assigned")
        return self._start

    @start.setter
    def start(self, val):
        if val < 0.0:
            raise ValueError("Negative start time is not valid")
        self._start = val

    @property
    def stop(self):
        if self._stop is None:
            raise RuntimeError("Stop time is not yet assigned")
        return self._stop

    @stop.setter
    def stop(self, val):
        if val < 0.0:
            raise ValueError("Negative stop time is not valid")
        self._stop = val

    @property
    def first(self):
        if self._first is None:
            raise RuntimeError("First sample is not yet assigned")
        return self._first

    @first.setter
    def first(self, val):
        if val < 0:
            raise ValueError("Negative first sample is not valid")
        self._first = val

    @property
    def last(self):
        if self._last is None:
            raise RuntimeError("Last sample is not yet assigned")
        return self._last

    @last.setter
    def last(self, val):
        if val < 0:
            raise ValueError("Negative last sample is not valid")
        self._last = val

    @property
    def range(self):
        b = self.start
        e = self.stop
        return (e - b)

    @property
    def samples(self):
        b = self.first
        e = self.last
        return (e - b + 1)
    
