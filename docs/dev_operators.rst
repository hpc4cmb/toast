.. _dev_ops:

Developing Operators
=======================

The ``Operator`` class is the building block for assembling workflows that simulate, reduce, or otherwise manipulate ``Data`` objects.

Overall Structure
-----------------------

Operators are classes which must have several required methods.  Operators must also be configured using class "traits" from the ``traitlets`` package.  Traits can be set at construction time or afterwards.  The constructor should take no additional parameters- everything should be a trait.  Using traits allows us to automate several things:

- Class docstring generated automatically from trait help strings.
- Class traits can be set from configuration files and / or commandline arguments.
- Class instances can be modified after construction by setting traits and there are mechanisms to validate trait values when the user sets them.

In addition to using traits for all configuration, each operator must define an ``_exec()`` method which accepts a ``Data`` instance (with one or more observations) and optionally a selection of detectors.  The operator can process the data however it likes (see below for guidelines of common patterns).  The operator must support the ``_exec()`` method being called multiple times, for example with different ``Data`` instances or different sets of detectors.

Each operator should also define a ``_finalize()`` method which will be called at the end of a sequence of calls to ``_exec()``.  This finalize method should perform any closing calculations needed.  If a call to ``_exec()`` is made **after** ``_finalize()``, the operator should reinitialize its state and prepare to start working again on new sets of data.

The Operator base class apply a helper method ``apply()``, which is a shortcut for running ``exec()`` and ``finalize()`` in a sequence.  It is useful when you only intend to make a single call to the exec method.

Operator classes typically require some input data fields in each observation and may create new output data as well.  The inputs in each observations should be returned by the ``_requires()`` method of the operator.  The outputs are returned by ``_provides()`` method.  Both of these return dictionaries of the required or provided names of metadata, shared, detdata and intervals data.

Over time, the traits used by an operator may change.  If an old configuration file is used to construct a "newer" version of the operator, the operator can simply raise an exception or it can attempt to translate this configuration.  Each configuration dumped from operator traits includes an "API" trait.  Each operator has this integer trait and can increment it whenever incompatible changes are made to its traits.  To support older versions of the API, the operator can define a ``translate()`` method that takes an older version of the configuration and translates it to the latest version of the configuration.

Data Processing Patterns
--------------------------------

Different data processing operations require passing through the data in different ways.  Additionally, there are cases when an operator will use other operators on subsets of data as part of the higher-level operation.  Here we go through some common patterns.

One Observation at a Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This pattern is used when there is an expensive (in terms of either calculation or memory use) operation that occurs within an observation and which can then be used by multiple detectors.  Some example operations like this might be simulating atmosphere that is common to all detectors or calculating planet locations for the timespan covered by an observation.

.. code-block:: python
    def _exec(self, data, detectors=None, **kwargs):
        for ob in data.obs:
            # (Some expensive operation here which is needed by all detectors.)
            # Get the detectors we are using for this observation
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue
            for det in dets:
                # (Do something for this detector)

This works fine for simple operators but what if we want use other, lower-level operators within our operator?  We can do this by calling the other operator's ``exec()`` and ``finalize()`` methods (or the shortcut ``apply()`` method) at the appropriate places.  First, imagine we want to run one operator before our code and another afterwards, for each observation:

.. code-block:: python
    def _exec(self, data, detectors=None, **kwargs):
        op_A = OperatorA()
        op_B = OperatorB()
        for iobs, ob in enumerate(data.obs):
            # Temporary data object with just this observation
            temp_data = data.select(obs_index=iobs)

            # Run op_A on all detectors for this observation
            op_A.apply(temp_data, detectors=detectors)

            # Now do our work on this observation.
            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue
            for det in dets:
                # (Do something for this detector)

            # Run op_B on all detectors for this observation
            op_B.apply(temp_data, detectors=detectors)

There are also times when we need to run an operator inside the "loop over detectors".  For example, if we are doing a small operation like computing the detector quaternion rotations from the boresight.  That scenario looks like this:

.. code-block:: python
    def _exec(self, data, detectors=None, **kwargs):
        op_A = OperatorA()
        op_B = OperatorB()
        for iobs, ob in enumerate(data.obs):
            # Temporary data object with just this observation
            temp_data = data.select(obs_index=iobs)

            dets = ob.select_local_detectors(detectors)
            if len(dets) == 0:
                # Nothing to do for this observation
                continue
            for det in dets:
                # Run op_A on this single observation and detector
                op_A.apply(temp_data, detectors=[det])

                # (Do something for this detector)

                # Run op_B on this single observation and detector
                op_B.apply(temp_data, detectors=[det])




One Detector at a Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This pattern is used when there is an expensive operation that occurs for each detector, and we want to make use of that result for all observations before moving on to the next detector.  An example would be simulating detector band-pass or beam convolution.

.. code-block:: python
    def _exec(self, data, detectors=None, **kwargs):
        # Get the superset of local detectors across all observations
        all_dets = data.all_local_detectors(selection=detectors)

        for det in all_dets:
            # Loop over all local detectors
            #
            # (Some expensive operation for this detector).
            #
            for ob in data.obs:
                # Loop over observations
                if det not in ob.local_detectors:
                    # This observation does not have this detector
                    continue
                # (Do something for this observation)
