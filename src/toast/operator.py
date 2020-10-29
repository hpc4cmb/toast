# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.


from .utils import Logger

from .traits import TraitConfig


class Operator(TraitConfig):
    """Base class for Operators.

    An operator has methods which work with a toast.dist.Data object.  This base class
    defines some interfaces and also some common helper methods.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, **kwargs):
        raise NotImplementedError("Fell through to Operator base class")

    def exec(self, data, detectors=None, **kwargs):
        """Perform operations on a Data object.

        If a list of detectors is specified, only process these detectors.  Any extra
        kwargs are passed to the derived class internal method.

        Args:
            data (toast.Data):  The distributed data.
            detectors (list):  A list of detector names or indices.  If None, this
                indicates a list of all detectors.

        Returns:
            None

        """
        self._exec(data, detectors=detectors, **kwargs)

    def _finalize(self, data, **kwargs):
        raise NotImplementedError("Fell through to Operator base class")

    def finalize(self, data, **kwargs):
        """Perform any final operations / communication.

        A call to this function indicates that all calls to the 'exec()' method are
        complete, and the operator should perform any final actions.  Any extra
        kwargs are passed to the derived class internal method.

        Args:
            data (toast.Data):  The distributed data.

        Returns:
            None

        """
        self._finalize(data, **kwargs)

    def apply(self, data, detectors=None, **kwargs):
        """Run exec() and finalize().

        This is a convenience wrapper that calls exec() exactly once with an optional
        detector list and then immediately calls finalize().  This is really only
        useful when working interactively to save a bit of typing.  When a `Pipeline`
        is calling other operators it will always use exec() and finalize() explicitly.

        After calling this, any future calls to exec() may produce unexpected results,
        since finalize() has already been called.

        Args:
            data (toast.Data):  The distributed data.
            detectors (list):  A list of detector names or indices.  If None, this
                indicates a list of all detectors.

        Returns:
            None

        """
        self.exec(data, detectors, **kwargs)
        self.finalize(data, **kwargs)

    def _requires(self):
        raise NotImplementedError("Fell through to Operator base class")
        return dict()

    def requires(self):
        """Dictionary of Observation keys directly used by this Operator.

        This dictionary should have 4 keys, each containing a list of "metadata",
        "detdata", "shared", and "intervals" fields.  Metadata keys are those contained
        in the primary observation dictionary.  Detdata, shared, and intervals keys are
        those contained in the "detdata", "shared", and "intervals" observation
        attributes.

        Returns:
            (dict):  The keys in the Observation dictionary required by the operator.

        """
        return self._requires()

    def _provides(self):
        raise NotImplementedError("Fell through to Operator base class")
        return dict()

    def provides(self):
        """Dictionary of Observation keys created by this Operator.

        This dictionary should have 4 keys, each containing a list of "metadata",
        "detdata", "shared", and "intervals" fields.  Metadata keys are those contained
        in the primary observation dictionary.  Detdata, shared, and intervals keys are
        those contained in the "detdata", "shared", and "intervals" observation
        attributes.

        Returns:
            (dict):  The keys in the Observation dictionary that will be created
                or modified.

        """
        return self._provides()

    def _accelerators(self):
        raise NotImplementedError("Fell through to Operator base class")
        return list()

    def accelerators(self):
        """List of accelerators supported by this Operator.

        Returns:
            (list):  List of pre-defined accelerator names supported by this
                operator (and by TOAST).

        """
        return self._accelerators()

    @classmethod
    def get_class_config_path(cls):
        return "/operators/{}".format(cls.__qualname__)

    def get_config_path(self):
        if self.name is None:
            return None
        return "/operators/{}".format(self.name)

    @classmethod
    def get_class_config(cls, input=None):
        """Return a dictionary of the default traits of an Operator class.

        This returns a new or appended dictionary.  The class instance properties are
        contained in a dictionary found in result["operators"][cls.name].

        If the specified named location in the input config already exists then an
        exception is raised.

        Args:
            input (dict):  The optional input dictionary to update.

        Returns:
            (dict):  The created or updated dictionary.

        """
        return super().get_class_config(section="operators", input=input)

    def get_config(self, input=None):
        """Return a dictionary of the current traits of an Operator *instance*.

        This returns a new or appended dictionary.  The operator instance properties are
        contained in a dictionary found in result["operators"][self.name].

        If the specified named location in the input config already exists then an
        exception is raised.

        Args:
            input (dict):  The optional input dictionary to update.

        Returns:
            (dict):  The created or updated dictionary.

        """
        return super().get_config(section="operators", input=input)

    @classmethod
    def translate(cls, props):
        """Given a config dictionary, modify it to match the current API."""
        # For operators, the derived classes should implement this method as needed
        # and then call super().translate(props) to trigger this method.  Here we strip
        # the 'API' key from the config.
        props = super().translate(props)
        if "API" in props:
            del props["API"]
        return props
