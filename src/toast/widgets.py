# Copyright (c) 2022-2022 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy
import re
from datetime import datetime, timedelta, timezone

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from IPython.display import clear_output, display
from plotly_resampler import FigureResampler

from .intervals import IntervalList
from .noise import Noise
from .observation import default_values as defaults
from .pixels import PixelData


class ObservationWidget(object):
    """Interactive widget for exploring observation data.

    Args:
        obs (Observation):  The observation to display.

    """

    not_selected = "(None)"

    def __init__(self, obs):
        self.obs = obs

        self._layout_box = widgets.Layout(
            justify_content="space-between",
            grid_gap="10px",
            padding="5px",
        )
        self._layout_bordered_box = widgets.Layout(
            border="solid 1px",
            justify_content="space-between",
            grid_gap="10px",
            padding="5px",
        )

        self.app = widgets.Tab()

        self.close_buttons = dict()

        self._observation_summary()
        display(self.app)

    def _key_value_table(
        self, title, data, key_name, val_name, height, width_percent=100
    ):
        """Create a 2-column table of values from a dictionary."""
        keys = list(data.keys())
        keydata = [str(data[x]) for x in keys]
        w_data = go.FigureWidget(
            data=[
                go.Table(
                    header=dict(values=[key_name, val_name]),
                    cells=dict(values=[keys, keydata]),
                )
            ],
            layout=go.Layout(
                title_text=title,
                margin=go.layout.Margin(
                    l=20,  # left margin
                    r=20,  # right margin
                    b=20,  # bottom margin
                    t=40,  # top margin
                ),
                height=height,
            ),
        )
        return widgets.Box(
            [w_data],
            layout=widgets.Layout(width=f"{width_percent}%", overflow_y="auto"),
        )

    def _timeselect_widget(self):
        """Create a widget that selects a timestamp field and zone."""
        time_list = list()
        for k in self.obs.shared.keys():
            sdata, scom = self.obs.shared._internal[k]
            sshape = sdata.shape
            if (scom == "column") and (
                len(sshape) == 1 or (len(sshape) == 2 and sshape[1] == 1)
            ):
                time_list.append(k)
        w_times = widgets.Dropdown(
            options=time_list,
            value=defaults.times,
            description="Time Field:",
            layout=widgets.Layout(width="90%"),
        )
        zones = [timezone(timedelta(hours=(x - 12))) for x in list(range(24))]
        tz_opts = [(x.tzname(None), y) for y, x in enumerate(zones)]
        w_zone = widgets.Dropdown(
            options=tz_opts,
            value=12,
            description="Displayed As:",
            layout=widgets.Layout(width="90%"),
        )

        def response_time(change):
            """Recompute the datetime labels when the time field or zone changes."""
            w_times.timedates = [
                datetime.fromtimestamp(x, tz=zones[w_zone.value])
                for x in self.obs.shared[w_times.value].data
            ]

        response_time(None)
        w_times.observe(response_time, names="value")
        w_zone.observe(response_time, names="value")
        return w_times, w_zone

    def _detselect_widget(self):
        """Create a widget that selects detectors."""
        det_names = ["ALL"]
        det_names.extend(self.obs.local_detectors)
        w_dets = widgets.SelectMultiple(
            options=det_names,
            value=[det_names[1]],
            description="Detectors:",
        )
        return w_dets

    def _interval_select_widget(self):
        """Create a widget that selects intervals."""
        inames = [self.not_selected]
        for k in self.obs.intervals.keys():
            if k == self.obs.intervals.all_name:
                continue
            inames.append(k)
        w_intr = widgets.SelectMultiple(
            options=inames,
            value=[inames[0]],
            description="Intervals:",
        )
        return w_intr

    def _close_tab(self, b):
        """Close a tab associated with a close button."""
        tabindex = self.close_buttons[b]
        if tabindex >= len(self.app.children):
            # Ignore the fake click when the button is created.
            return
        current = self.app.children
        current_titles = {x: y for x, y in enumerate(self.app.titles)}
        new_children = list()
        new_titles = dict()
        for tb in range(0, tabindex):
            new_children.append(current[tb])
            new_titles[tb] = current_titles[tb]
        for tb in range(tabindex + 1, len(current)):
            new_children.append(current[tb])
            new_titles[tb - 1] = current_titles[tb]
        del current
        del self.close_buttons[b]
        for k in list(self.close_buttons.keys()):
            if self.close_buttons[k] > tabindex:
                self.close_buttons[k] -= 1
        self.app.children = new_children
        self.app.titles = [new_titles[tb] for tb in range(len(self.app.children))]

    def _detdata_display(self, name):
        """Create a detector data display widget."""

        # Create a button to close this tab
        this_tab = len(self.app.children)
        close_button = widgets.Button(
            description="Close",
            button_style="danger",
            tooltip=f"Close",
            layout=widgets.Layout(width="10%"),
        )
        close_button.on_click(self._close_tab)
        self.close_buttons[close_button] = this_tab

        w_times, w_zone = self._timeselect_widget()
        tz_layout = widgets.Layout(
            justify_content="space-between",
            grid_gap="10px",
            padding="5px",
            width="30%",
        )
        w_timezone = widgets.VBox(
            [w_times, w_zone],
            layout=self._layout_box,
        )

        sel_layout = widgets.Layout(width="30%")
        w_dets = self._detselect_widget()
        w_dets.layout = sel_layout
        w_intr = self._interval_select_widget()
        w_intr.layout = sel_layout

        w_topbar = widgets.HBox(
            children=[w_timezone, w_dets, w_intr, close_button],
            layout=self._layout_box,
        )

        fig = FigureResampler(go.Figure())
        w_plot = widgets.Output()

        def response(change):
            det_list = list()
            if w_dets.value[0] == "ALL":
                det_list.extend(self.obs.local_detectors)
            else:
                det_list.extend(w_dets.value)
            # To save plotting time, pre-compute the union of selected intervals
            plot_intr = None
            for ilist in w_intr.value:
                if ilist == self.not_selected:
                    continue
                if plot_intr is None:
                    plot_intr = IntervalList(
                        self.obs.shared[w_times.value].data,
                        intervals=self.obs.intervals[ilist],
                    )
                else:
                    plot_intr |= self.obs.intervals[ilist]

            fig.data = list()
            fig.layout["shapes"] = list()

            # with w_plot.batch_update():
            for det in det_list:
                fig.add_trace(
                    go.Scatter(
                        mode="lines",
                        name=det,
                    ),
                    hf_x=w_times.timedates,
                    hf_y=self.obs.detdata[name][det].astype(np.float32),
                )
            intr_shapes = dict()
            if plot_intr is not None:
                plot_intr.simplify()
                for intr in plot_intr:
                    begin = intr.first
                    end = intr.last
                    shp_name = f"interval_{begin}"
                    intr_shapes[shp_name] = go.layout.Shape(
                        type="rect",
                        x0=w_times.timedates[begin],
                        x1=w_times.timedates[end],
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="y domain",
                        line_width=0,
                        fillcolor="gray",
                        opacity=0.3,
                    )
            fig.update_layout(
                title=dict(text=f"Detector Data ({name})"),
                barmode="overlay",
                xaxis_title="Time",
                xaxis_tickformat="%H:%M:%S<br>%m-%d<br>%Y",
                yaxis_title="Detector Value",
                margin=go.layout.Margin(
                    l=10,  # left margin
                    r=10,  # right margin
                    b=20,  # bottom margin
                    t=40,  # top margin
                ),
                shapes=list(intr_shapes.values()),
            )
            with w_plot:
                clear_output(wait=True)
                fig.show_dash(mode="inline", host="localhost")

        response(None)
        w_times.observe(response, names="value")
        w_zone.observe(response, names="value")
        w_dets.observe(response, names="value")
        w_intr.observe(response, names="value")

        new_tab = widgets.VBox(
            [
                w_topbar,
                w_plot,
            ],
            layout=widgets.Layout(border="solid 1px"),
        )

        new_children = list(self.app.children)
        new_children.append(new_tab)
        self.app.children = new_children
        self.app.set_title(len(self.app.children) - 1, f"{name}")

    def _new_plot(self, b):
        """Create the proper type of plot"""
        if self.create_plot_select.value == self.not_selected:
            return
        # FIXME:  call shared vs shared plot here if Y is a shared field
        # Extract detdata name
        det_pat = re.compile(r"Detector (.*)")
        det_mat = det_pat.match(self.create_plot_select.value)
        if det_mat is not None:
            detdata_name = det_mat.group(1)
            self._detdata_display(detdata_name)

    def _plot_create(self):
        """Create a new plot activation."""
        # Allowable X axis vectors are shared data with one value per
        # sample.
        # self.x_list = [self.not_selected]
        # shared_list = list()
        # for k in self.obs.shared.keys():
        #     sdata, scom = self.obs.shared._internal[k]
        #     sshape = sdata.shape
        #     if (scom == "column") and (len(sshape) == 1 or (len(sshape) == 2 and sshape[1] == 1)):
        #         shared_list.append(k)
        # self.x_list.extend(shared_list)
        # For Y, we can plot the same shared data plus single-valued detector data.
        self.obj_list = [self.not_selected]
        for k in self.obs.detdata.keys():
            dshape = self.obs.detdata[k].detector_shape
            if len(dshape) == 1 or (len(dshape) == 2 and dshape[1] == 1):
                self.obj_list.append(f"Detector {k}")

        # FIXME:  Restore this once we can plot shared vs shared
        # self.y_list.extend(shared_list)

        self.create_plot_select = widgets.Dropdown(
            options=self.obj_list,
            value=self.not_selected,
            description="Data:",
        )

        self.create_button = widgets.Button(
            description="Create",
            button_style="primary",
            tooltip=f"Create",
            layout=widgets.Layout(width=f"25%"),
        )
        self.create_button.on_click(self._new_plot)

        w_create_plot = widgets.HBox(
            [
                widgets.Label(value="New Plot:"),
                self.create_plot_select,
                self.create_button,
            ],
            layout=self._layout_bordered_box,
        )
        return w_create_plot

    def _observation_summary(self):
        """Create the observation summary tab."""
        header = widgets.HBox(
            [
                widgets.Label(value=f'  Name: "{self.obs.name}"'),
                widgets.Label(value=f"UID: {self.obs.uid}"),
                widgets.Label(value=f"Samples: {self.obs.n_all_samples}"),
                widgets.Label(value="  "),
            ],
            layout=self._layout_bordered_box,
        )
        info = [header]

        tele = self.obs.telescope
        fp = tele.focalplane
        tele_info = {
            "Name": tele.name,
            "UID": tele.uid,
            "Site": tele.site.name,
            "Focalplane": f"{len(fp.detector_data)} detectors at {fp.sample_rate:0.1f}",
        }
        w_tele = self._key_value_table(
            "Telescope", tele_info, "Property", "Value", 300, width_percent=100
        )

        meta_info = dict()
        for k, v in self.obs.items():
            meta_info[k] = str(v)
        w_meta = self._key_value_table(
            "Metadata", meta_info, "Property", "Value", 300, width_percent=100
        )

        w_telemeta = widgets.VBox([w_tele, w_meta], layout=widgets.Layout(width="40%"))

        shared_info = dict()
        for k in self.obs.shared.keys():
            sdata, comtype = self.obs.shared._internal[k]
            shared_info[k] = f"{sdata.shape} of {sdata.dtype} (comm={comtype})"
        w_shared = self._key_value_table(
            "Shared Data", shared_info, "Field", "Info", 200, width_percent=100
        )

        det_info = dict()
        for k in self.obs.detdata.keys():
            ddata = self.obs.detdata[k]
            ndet = len(ddata.detectors)
            det_info[
                k
            ] = f"{ndet} dets: {ddata.shape} of {ddata.dtype} (unit='{ddata.units}')"
        w_detdata = self._key_value_table(
            "Detector Data", det_info, "Field", "Info", 200, width_percent=100
        )

        intr_info = dict()
        for k in self.obs.intervals.keys():
            if k == self.obs.intervals.all_name:
                continue
            idata = self.obs.intervals[k]
            nintr = len(idata)
            intr_info[k] = f"{nintr} spans"
        w_intr = self._key_value_table(
            "Intervals", intr_info, "Name", "Info", 200, width_percent=100
        )

        w_datacol = widgets.VBox(
            [w_shared, w_detdata, w_intr], layout=widgets.Layout(width="60%")
        )

        info.append(
            widgets.HBox(
                [
                    w_telemeta,
                    w_datacol,
                ],
            )
        )

        # Plot creation
        info.append(self._plot_create())

        new_children = list(self.app.children)
        new_children.append(
            widgets.VBox(
                info,
            )
        )
        self.app.children = new_children
        self.app.set_title(len(self.app.children) - 1, "Observation")
