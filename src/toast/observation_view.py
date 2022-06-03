# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numbers
import sys
from collections.abc import Mapping, MutableMapping, Sequence

import numpy as np


class DetDataView(MutableMapping):
    """Class that applies views to a DetDataManager instance."""

    def __init__(self, obj, slices):
        self.obj = obj
        self.slices = slices

    # Mapping methods

    def __getitem__(self, key):
        vw = [self.obj.detdata[key].view((slice(None), x)) for x in self.slices]
        return vw

    def __delitem__(self, key):
        raise RuntimeError(
            "Cannot delete views of detdata, since they are created on demand"
        )

    def __setitem__(self, key, value):
        vw = [self.obj.detdata[key].view((slice(None), x)) for x in self.slices]
        if isinstance(value, numbers.Number) or len(value) == 1:
            # This is a numerical scalar or identical array for all slices
            for v in vw:
                v[:] = value
        else:
            # One element of value for each slice
            if len(value) != len(vw):
                msg = "when assigning to a view, you must have one value or one value for each view interval"
                raise RuntimeError(msg)
            vw[:] = value

    def __iter__(self):
        return iter(self.slices)

    def __len__(self):
        return len(self.slices)

    def __repr__(self):
        val = "<DetDataView {} slices".format(len(self.slices))
        val += ">"
        return val


class SharedView(MutableMapping):
    """Class that applies views to a SharedDataManager instance."""

    def __init__(self, obj, slices):
        self.obj = obj
        self.slices = slices

    # Mapping methods

    def __getitem__(self, key):
        vw = [self.obj.shared[key][x] for x in self.slices]
        return vw

    def __delitem__(self, key):
        raise RuntimeError(
            "Cannot delete views of shared data, since they are created on demand"
        )

    def __setitem__(self, key, value):
        raise RuntimeError(
            "Cannot set views of shared data- use the set() method on the original."
        )

    def __iter__(self):
        return iter(self.slices)

    def __len__(self):
        return len(self.slices)

    def __repr__(self):
        val = "<SharedView {} slices".format(len(self.slices))
        val += ">"
        return val


class View(Sequence):
    """Class representing a list of views into any of the local observation data."""

    def __init__(self, obj, key):
        self.obj = obj
        self.key = key
        # Compute a list of slices for these intervals
        self.slices = [slice(x.first, x.last + 1, 1) for x in self.obj.intervals[key]]
        self.detdata = DetDataView(obj, self.slices)
        self.shared = SharedView(obj, self.slices)

    def __getitem__(self, key):
        return self.slices[key]

    def __contains__(self, item):
        for sl in self.slices:
            if sl == item:
                return True
        return False

    def __iter__(self):
        return iter(self.slices)

    def __len__(self):
        return len(self.slices)

    def __repr__(self):
        s = "["
        if len(self.slices) > 1:
            for it in self.slices[0:-1]:
                s += str(it)
                s += ", "
        if len(self.slices) > 0:
            s += str(self.slices[-1])
        s += "]"
        return s


class ViewManager(MutableMapping):
    """Internal class to manage views into observation data objects."""

    def __init__(self, obj):
        self.obj = obj
        if not hasattr(obj, "_views"):
            self.obj._views = dict()

    # Mapping methods

    def __getitem__(self, key):
        view_name = key
        if view_name is None:
            # Make sure the fake internal intervals are created
            trigger = self.obj.intervals[None]
            view_name = self.obj.intervals.all_name
        if view_name not in self.obj._views:
            # View does not yet exist, create it.
            if key is not None and key not in self.obj.intervals:
                raise KeyError(
                    "Observation does not have interval list named '{}'".format(key)
                )
            self.obj._views[view_name] = View(self.obj, key)
            # Register deleter callback
            if key is not None:
                self.obj.intervals.register_delete_callback(key, self.__delitem__)
        return self.obj._views[view_name]

    def __delitem__(self, key):
        del self.obj._views[key]

    def __setitem__(self, key, value):
        raise RuntimeError("Cannot set views directly- simply access them.")

    def __iter__(self):
        return iter(self.obj)

    def __len__(self):
        return len(self.obj)

    def clear(self):
        self.obj._views.clear()


class ViewInterface(object):
    """Descriptor class for accessing the views in an observation.

    You can get a view of the data for a particular interval list just by accessing
    it with the name of the intervals object you want:

    obs.view["name_of_intervals"]

    Then you can use this to provide a view into either detdata or shared objects within
    the observation.  For example:

    print(obs.view["name_of_intervals"].detdata["signal"])

    obs.view["bad_pointing"].shared["boresight"][:] = np.array([0., 0., 0., 1.])

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        else:
            if not hasattr(obj, "_viewmgr"):
                obj._viewmgr = ViewManager(obj)
            return obj._viewmgr

    def __set__(self, obj, value):
        raise AttributeError("Cannot reset the view interface")

    def __delete__(self, obj):
        raise AttributeError("Cannot delete the view interface")
