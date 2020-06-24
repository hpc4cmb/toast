# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


from .utils import Logger

# Detect whether we are using pyCUDA

use_pycuda = None
cuda = None
cuda_devices = 0

if use_pycuda is None:
    try:
        import pycuda.driver as cuda

        cuda.init()
        n_dev = cuda.Device.count()
        if n_dev > 0:
            use_pycuda = True
            cuda_devices = n_dev
        else:
            use_pycuda = False
    except:
        use_pycuda = False


class AcceleratorCuda(object):
    """Class storing the device properties, context, and streams for one process.
    """

    def __init__(self, device_index):
        self._device_index = device_index
        self._device = cuda.Device(device_index)
        self._device_attr = self._device.get_attributes()
        self._context = self._device.make_context()
        self._streams = dict()

    def close(self):
        """Explicitly shut down the context and streams.
        """
        if hasattr(self, "_streams") and self._streams is not None:
            for k, v in self._streams.items():
                pass
            self._streams = None
        if hasattr(self, "_context") and self._context is not None:
            self._context.pop()
            self._context = None

    def __del__(self):
        self.close()

    @property
    def device(self):
        """The cuda.Device
        """
        return self._device

    @property
    def device_index(self):
        """The cuda.Device index
        """
        return self._device_index

    @property
    def device_attr(self):
        """The cuda.Device attributes
        """
        return self._device_attr

    @property
    def context(self):
        """The Context on this device
        """
        return self._context

    def get_stream(self, name):
        """Get the stream with the specified name.

        This creates the stream if it does not exist.

        Args:
            name (str):  The name of the stream.

        Returns:
            (Stream):  The cuda stream.

        """
        if name not in self._streams:
            self._streams[name] = cuda.Stream()

        return self._streams[name]

    def del_stream(self, name):
        """Delete the specified stream.

        This performs a sync on the stream and then removes it.

        Args:
            name (str):  The name of the stream.

        Returns:
            None

        """
        if name not in self._streams:
            return
        self._streams[name].synchronize()
        del self._streams[name]
