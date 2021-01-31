# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

import traitlets

from ..utils import Logger

from ..traits import TraitConfig, Instance, Unicode, Int

from ..data import Data


class Template(TraitConfig):
    """Base class for timestream templates.

    A template defines a mapping to / from timestream values to a set of template
    amplitudes.  These amplitudes are usually quantities being solved as part of the
    map-making.  Examples of templates might be destriping baseline offsets,
    azimuthally binned ground pickup, etc.

    The template amplitude data may be distributed in a variety of ways.  For some
    types of templates, every process may have their own unique set of amplitudes based
    on the data that they have locally.  In other cases, every process may have a full
    local copy of all template amplitudes.  There might also be cases where each
    process has a non-unique subset of amplitude values (similar to the way that
    pixel domain quantities are distributed).

    """

    # Note:  The TraitConfig base class defines a "name" attribute.

    data = Instance(
        klass=Data,
        allow_none=True,
        help="This must be an instance of a Data class (or None)",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    flags = Unicode(
        None, allow_none=True, help="Observation detdata key for solver flags to use"
    )

    flag_mask = Int(0, help="Bit mask value for solver flags")

    @traitlets.validate("data")
    def _check_data(self, proposal):
        dat = proposal["value"]
        if dat is not None:
            if not isinstance(dat, Data):
                raise traitlets.TraitError("data should be a Data instance")
        return dat

    @traitlets.observe("data")
    def initialize(self, change):
        newdata = change["new"]
        self._initialize(newdata)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, new_data):
        # Derived classes should implement this method to do any set up (like
        # computing the number of amplitudes) whenever the data changes.
        raise NotImplementedError("Derived class must implement _initialize()")

    def _detectors(self):
        # Derived classes should return the list of detectors they support.
        raise NotImplementedError("Derived class must implement _detectors()")

    def detectors(self):
        """Return a list of detectors supported by the template.

        This list will change whenever the `data` trait is set, which initializes
        the template.

        Returns:
            (list):  The detectors with local amplitudes across all observations.

        """
        if self.data is None:
            raise RuntimeError("You must set the data trait before calling detectors()")
        return self._detectors()

    def _zeros(self):
        raise NotImplementedError("Derived class must implement _zeros()")

    def zeros(self):
        """Return an Amplitudes object filled with zeros.

        This returns an Amplitudes instance with appropriate dimensions for this
        template.  This will raise an exception if called before the `data` trait
        is set.

        Returns:
            (Amplitudes):  Zero amplitudes.

        """
        if self.data is None:
            raise RuntimeError("You must set the data trait before using a template")
        return self._zeros()

    def _add_to_signal(self, detector, amplitudes):
        raise NotImplementedError("Derived class must implement _add_to_signal()")

    def add_to_signal(self, detector, amplitudes):
        """Accumulate the projected amplitudes to a timestream.

        This performs the operation:

        .. math::
            s += F \\cdot a

        Where `s` is the det_data signal, `F` is the template and `a` is the amplitudes.

        Args:
            detector (str):  The detector name.
            amplitudes (Amplitudes):  The Amplitude values for this template.

        Returns:
            None

        """
        if self.data is None:
            raise RuntimeError("You must set the data trait before using a template")
        return self._add_to_signal(detector, amplitudes)

    def _project_signal(self, detector, amplitudes):
        raise NotImplementedError("Derived class must implement _project_signal()")

    def project_signal(self, detector, amplitudes):
        """Project a timestream into template amplitudes.

        This performs:

        .. math::
            a += F^T \\cdot s

        Where `s` is the det_data signal, `F` is the template and `a` is the amplitudes.

        Args:
            detector (str):  The detector name.
            amplitudes (Amplitudes):  The Amplitude values for this template.

        Returns:
            None

        """
        if self.data is None:
            raise RuntimeError("You must set the data trait before using a template")
        self._project_signal(detector, amplitudes)

    def _add_prior(self, amplitudes_in, amplitudes_out):
        # Not all Templates implement the prior
        return

    def add_prior(self, amplitudes_in, amplitudes_out):
        """Apply the inverse amplitude covariance as a prior.

        This performs:

        .. math::
            a' += {C_a}^{-1} \\cdot a

        Args:
            amplitudes_in (Amplitudes):  The input Amplitude values for this template.
            amplitudes_out (Amplitudes):  The input Amplitude values for this template.

        Returns:
            None

        """
        if self.data is None:
            raise RuntimeError("You must set the data trait before using a template")
        self._add_prior(amplitudes_in, amplitudes_out)

    def _apply_precond(self, amplitudes_in, amplitudes_out):
        raise NotImplementedError("Derived class must implement _apply_precond()")

    def apply_precond(self, amplitudes_in, amplitudes_out):
        """Apply the template preconditioner.

        Formally, the preconditioner "M" is an approximation to the "design matrix"
        (the "A" matrix in "Ax = b").  This function applies the inverse preconditioner
        to the template amplitudes:

        .. math::
            a' += M^{-1} \\cdot a

        Args:
            amplitudes_in (Amplitudes):  The input Amplitude values for this template.
            amplitudes_out (Amplitudes):  The input Amplitude values for this template.

        Returns:
            None

        """
        if self.data is None:
            raise RuntimeError("You must set the data trait before using a template")
        self._apply_precond(amplitudes_in, amplitudes_out)

    def _accelerators(self):
        # Do not force descendent classes to implement this.  If it is not
        # implemented, then it is clear that the class does not support any
        # accelerators
        return list()

    def accelerators(self):
        """List of accelerators supported by this Template.

        Returns:
            (list):  List of pre-defined accelerator names supported by this
                operator (and by TOAST).

        """
        return self._accelerators()

    @classmethod
    def get_class_config_path(cls):
        return "/templates/{}".format(cls.__qualname__)

    def get_config_path(self):
        if self.name is None:
            return None
        return "/templates/{}".format(self.name)

    @classmethod
    def get_class_config(cls, input=None):
        """Return a dictionary of the default traits of an Template class.

        This returns a new or appended dictionary.  The class instance properties are
        contained in a dictionary found in result["templates"][cls.name].

        If the specified named location in the input config already exists then an
        exception is raised.

        Args:
            input (dict):  The optional input dictionary to update.

        Returns:
            (dict):  The created or updated dictionary.

        """
        return super().get_class_config(section="templates", input=input)

    def get_config(self, input=None):
        """Return a dictionary of the current traits of a Template *instance*.

        This returns a new or appended dictionary.  The operator instance properties are
        contained in a dictionary found in result["templates"][self.name].

        If the specified named location in the input config already exists then an
        exception is raised.

        Args:
            input (dict):  The optional input dictionary to update.

        Returns:
            (dict):  The created or updated dictionary.

        """
        return super().get_config(section="templates", input=input)

    @classmethod
    def translate(cls, props):
        """Given a config dictionary, modify it to match the current API."""
        # For templates, the derived classes should implement this method as needed
        # and then call super().translate(props) to trigger this method.  Here we strip
        # the 'API' key from the config.
        props = super().translate(props)
        if "API" in props:
            del props["API"]
        return props
