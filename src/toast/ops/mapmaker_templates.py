# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections import OrderedDict

import traitlets

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool, List

from ..timing import function_timer

from ..templates import Template, AmplitudesMap

from .operator import Operator


@trait_docs
class TemplateMatrix(Operator):
    """Operator for projecting or accumulating template amplitudes."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    templates = List(
        None, allow_none=True, help="This should be a list of Template instances"
    )

    amplitudes = Unicode(None, allow_none=True, help="Data key for template amplitudes")

    transpose = Bool(False, help="If True, apply the transpose.")

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

    @traitlets.validate("templates")
    def _check_templates(self, proposal):
        temps = proposal["value"]
        if temps is None:
            return temps
        for tp in temps:
            if not isinstance(tp, Template):
                raise traitlets.TraitError(
                    "templates must be a list of Template instances or None"
                )
        return temps

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialized = False

    def duplicate(self):
        """Make a shallow copy which contains the same list of templates.

        This is useful when we want to use both a template matrix and its transpose
        in the same pipeline.

        Returns:
            (TemplateMatrix):  A new instance with the same templates.

        """
        ret = TemplateMatrix(
            API=self.API,
            templates=self.templates,
            amplitudes=self.amplitudes,
            transpose=self.transpose,
            view=self.view,
            det_data=self.det_data,
            flags=self.flags,
            flag_mask=self.flag_mask,
        )
        ret._initialized = self._initialized
        return ret

    def apply_precond(self, amps_in, amps_out):
        """Apply the preconditioner from all templates to the amplitudes.

        This can only be called after the operator has been used at least once so that
        the templates are initialized.

        Args:
            amps_in (AmplitudesMap):  The input amplitudes.
            amps_out (AmplitudesMap):  The output amplitudes, modified in place.

        Returns:
            None

        """
        if not self._initialized:
            raise RuntimeError(
                "You must call exec() once before applying preconditioners"
            )
        for tmpl in self.templates:
            tmpl.apply_precond(amps_in[tmpl.name], amps_out[tmpl.name])

    def add_prior(self, amps_in, amps_out):
        """Apply the noise prior from all templates to the amplitudes.

        This can only be called after the operator has been used at least once so that
        the templates are initialized.

        Args:
            amps_in (AmplitudesMap):  The input amplitudes.
            amps_out (AmplitudesMap):  The output amplitudes, modified in place.

        Returns:
            None

        """
        if not self._initialized:
            raise RuntimeError(
                "You must call exec() once before applying the noise prior"
            )
        for tmpl in self.templates:
            tmpl.add_prior(amps_in[tmpl.name], amps_out[tmpl.name])

    @function_timer
    def _exec(self, data, detectors=None, use_accel=False, **kwargs):
        log = Logger.get()

        # Check that the detector data is set
        if self.det_data is None:
            raise RuntimeError("You must set the det_data trait before calling exec()")

        # Check that amplitudes is set
        if self.amplitudes is None:
            raise RuntimeError(
                "You must set the amplitudes trait before calling exec()"
            )

        # On the first call, we initialize all templates using the Data instance and
        # the fixed options for view, flagging, etc.
        if not self._initialized:
            for tmpl in self.templates:
                tmpl.view = self.view
                tmpl.flags = self.flags
                tmpl.flag_mask = self.flag_mask
                # This next line will trigger calculation of the number
                # of amplitudes within each template.
                tmpl.data = data
            self._initialized = True

        # Set template accelerator use
        for tmpl in self.templates:
            tmpl.use_accel = use_accel

        # Set the data we are using for this execution
        for tmpl in self.templates:
            tmpl.det_data = self.det_data

        # We loop over detectors.  Internally, each template loops over observations
        # and ignores observations where the detector does not exist.

        all_dets = data.all_local_detectors(selection=detectors)

        if self.transpose:
            if self.amplitudes not in data:
                # The output template amplitudes do not yet exist.  Create these with
                # all zero values.
                data[self.amplitudes] = AmplitudesMap()
                for tmpl in self.templates:
                    data[self.amplitudes][tmpl.name] = tmpl.zeros()
                if use_accel:
                    data[self.amplitudes].accel_create()
            for d in all_dets:
                for tmpl in self.templates:
                    tmpl.project_signal(d, data[self.amplitudes][tmpl.name])
        else:
            if self.amplitudes not in data:
                msg = "Template amplitudes '{}' do not exist in data".format(
                    self.amplitudes
                )
                log.error(msg)
                raise RuntimeError(msg)
            # Ensure that our output detector data exists in each observation
            for ob in data.obs:
                # Get the detectors we are using for this observation
                dets = ob.select_local_detectors(selection=detectors)
                if len(dets) == 0:
                    # Nothing to do for this observation
                    continue
                exists = ob.detdata.ensure(self.det_data, detectors=dets)
                for d in dets:
                    ob.detdata[self.det_data][d, :] = 0
                log.verbose(
                    f"TemplateMatrix {ob.name}:  input host detdata={ob.detdata[self.det_data][:][0:10]}"
                )
                if use_accel:
                    if not exists and not ob.detdata.accel_present(self.det_data):
                        ob.detdata.accel_create(self.det_data)

            for d in all_dets:
                for tmpl in self.templates:
                    log.verbose(f"TemplateMatrix {d} add to signal {tmpl.name}")
                    tmpl.add_to_signal(d, data[self.amplitudes][tmpl.name])
        return

    def _finalize(self, data, use_accel=False, **kwargs):
        if self.transpose:
            # Synchronize the result
            if use_accel:
                data[self.amplitudes].accel_update_host()
            for tmpl in self.templates:
                data[self.amplitudes][tmpl.name].sync()
            if use_accel:
                data[self.amplitudes].accel_update_device()
        return

    def _requires(self):
        req = {
            "global": list(),
            "meta": list(),
            "shared": list(),
            "detdata": list(),
            "intervals": list(),
        }
        if self.view is not None:
            req["intervals"].append(self.view)
        if self.transpose:
            req["detdata"].append(self.det_data)
            if self.shared_flags is not None:
                req["shared"].append(self.shared_flags)
            if self.det_flags is not None:
                req["detdata"].append(self.det_flags)
        else:
            req["global"].append(self.amplitudes)
        return req

    def _provides(self):
        prov = dict()
        if self.transpose:
            prov["global"] = [self.amplitudes]
        else:
            prov["detdata"] = [self.det_data]
        return prov

    def _supports_accel(self):
        # This is a logical AND of our templates
        for tmpl in self.templates:
            if not tmpl.supports_accel():
                return False
        return True
