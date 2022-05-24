# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from ..timing import function_timer_stackskip
from ..traits import TraitConfig
from ..utils import Logger


class Operator(TraitConfig):
    """Base class for Operators.

    An operator has methods which work with a toast.dist.Data object.  This base class
    defines some interfaces and also some common helper methods.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _exec(self, data, detectors=None, **kwargs):
        raise NotImplementedError("Fell through to Operator base class")

    @function_timer_stackskip
    def exec(self, data, detectors=None, use_accel=False, **kwargs):
        """Perform operations on a Data object.

        If a list of detectors is specified, only process these detectors.  Any extra
        kwargs are passed to the derived class internal method.

        Accelerator use:  If the derived class supports OpenACC and all the required
        data objects exist on the device, then the `_exec()` method will be called
        with the "use_accel=True" option.  Any operator that returns "True" from its
        _supports_accel() method should also accept the "use_accel" keyword argument.

        Args:
            data (toast.Data):  The distributed data.
            detectors (list):  A list of detector names or indices.  If None, this
                indicates a list of all detectors.

        Returns:
            None

        """
        log = Logger.get()
        if self.enabled:
            if use_accel and not self.accel_have_requires(data):
                msg = f"Operator {self.name} exec: required inputs not on device. "
                msg += "There is likely a problem with a Pipeline or the "
                msg += "operator dependencies."
                raise RuntimeError(msg)
            self._exec(
                data,
                detectors=detectors,
                use_accel=use_accel,
                **kwargs,
            )
        else:
            if data.comm.world_rank == 0:
                msg = f"Operator {self.name} is disabled, skipping call to exec()"
                log.debug(msg)

    def _finalize(self, data, **kwargs):
        raise NotImplementedError("Fell through to Operator base class")

    @function_timer_stackskip
    def finalize(self, data, use_accel=False, **kwargs):
        """Perform any final operations / communication.

        A call to this function indicates that all calls to the 'exec()' method are
        complete, and the operator should perform any final actions.  Any extra
        kwargs are passed to the derived class internal method.

        Args:
            data (toast.Data):  The distributed data.

        Returns:
            (value):  None or an Operator-dependent result.

        """
        log = Logger.get()
        if self.enabled:
            if use_accel and not self.accel_have_requires(data):
                msg = f"Operator {self.name} finalize: required inputs not on device. "
                msg += "There is likely a problem with a Pipeline or the "
                msg += "operator dependencies."
                raise RuntimeError(msg)
            return self._finalize(data, use_accel=use_accel, **kwargs)
        else:
            if data.comm.world_rank == 0:
                msg = f"Operator {self.name} is disabled, skipping call to finalize()"
                log.debug(msg)

    @function_timer_stackskip
    def apply(self, data, detectors=None, use_accel=False, **kwargs):
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
            (value):  None or an Operator-dependent result.

        """
        self.exec(data, detectors, use_accel=use_accel, **kwargs)
        return self.finalize(data, use_accel=use_accel, **kwargs)

    def _requires(self):
        raise NotImplementedError("Fell through to Operator base class")
        return dict()

    def requires(self):
        """Dictionary of Observation keys directly used by this Operator.

        This dictionary should have 5 keys, each containing a list of "global",
        "metadata", "detdata", "shared", and "intervals" fields.  Global keys are
        contained in the top-level data object.  Metadata keys are those contained
        in the primary observation dictionary.  Detdata, shared, and intervals keys are
        those contained in the "detdata", "shared", and "intervals" observation
        attributes.

        Returns:
            (dict):  The keys in the Observation dictionary required by the operator.

        """
        # Ensure that all keys exist
        req = self._requires()
        for key in ["global", "meta", "detdata", "shared", "intervals"]:
            if key not in req:
                req[key] = list()
        # All operators use an implied interval list of the full sample range
        if None not in req["intervals"]:
            req["intervals"].append(None)
        return req

    def _provides(self):
        raise NotImplementedError("Fell through to Operator base class")
        return dict()

    def provides(self):
        """Dictionary of Observation keys created by this Operator.

        This dictionary should have 5 keys, each containing a list of "global",
        "metadata", "detdata", "shared", and "intervals" fields.  Global keys are
        contained in the top-level data object.  Metadata keys are those contained
        in the primary observation dictionary.  Detdata, shared, and intervals keys are
        those contained in the "detdata", "shared", and "intervals" observation
        attributes.

        Returns:
            (dict):  The keys in the Observation dictionary that will be created
                or modified.

        """
        # Ensure that all keys exist
        prov = self._provides()
        for key in ["global", "meta", "detdata", "shared", "intervals"]:
            if key not in prov:
                prov[key] = list()
        return prov

    def _supports_accel(self):
        return False

    def supports_accel(self):
        """Query whether the operator supports OpenACC

        Returns:
            (bool):  True if the operator can use OpenACC, else False.

        """
        return self._supports_accel()

    def accel_have_requires(self, data):
        # Helper function to determine if all requirements are met to use accelerator
        # for a data object.
        log = Logger.get()
        if not self.supports_accel():
            # No support for OpenACC
            return False
        all_present = True
        req = self.requires()
        for ob in data.obs:
            for key in req["detdata"]:
                if not ob.detdata.accel_exists(key):
                    msg = f"{self.name}:  obs {ob.name}, detdata {key} not on device"
                    all_present = False
                else:
                    msg = f"{self.name}:  obs {ob.name}, detdata {key} is on device"
                log.verbose(msg)
            for key in req["shared"]:
                if not ob.shared.accel_exists(key):
                    msg = f"{self.name}:  obs {ob.name}, shared {key} not on device"
                    all_present = False
                else:
                    msg = f"{self.name}:  obs {ob.name}, shared {key} is on device"
                log.verbose(msg)
            for key in req["intervals"]:
                if not ob.intervals.accel_exists(key):
                    msg = f"{self.name}:  obs {ob.name}, intervals {key} not on device"
                    all_present = False
                else:
                    msg = f"{self.name}:  obs {ob.name}, intervals {key} is on device"
                log.verbose(msg)
        if all_present:
            log.verbose(f"{self.name}:  obs {ob.name} all required inputs on device")
        else:
            log.verbose(
                f"{self.name}:  obs {ob.name} some required inputs not on device"
            )
        return all_present

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
