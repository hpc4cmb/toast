#!/usr/bin/env python3

# Copyright (c) 2021-2021 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

"""Plot global timing results
"""

import argparse
import csv
import re
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio


def main():
    parser = argparse.ArgumentParser(description="Plot a timing dump.")

    parser.add_argument(
        "--file",
        required=True,
        help="Name of input CSV file",
    )

    parser.add_argument(
        "--out",
        required=False,
        default="timing.pdf",
        help="Name of output file",
    )

    args = parser.parse_args()

    if re.match(r".*\.pdf", args.out) is None:
        print("WARNING:  output file should have a .pdf extension")

    raw = dict()

    with open(args.file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            name = row[0]
            mat = re.match(r"\(function\) (.*)", name)
            if mat is not None:
                # this is a function timer
                trace = mat.group(1)
                value = float(row[7])
                if trace == "main":
                    # special case
                    continue
                else:
                    levels = trace.split("|")
                    parent = raw
                    for lvl in levels:
                        if lvl not in parent:
                            parent[lvl] = dict()
                        parent = parent[lvl]
                    parent["TOTAL"] = value

    # Prune lone branches and compute child node subtotals

    def process(node, name=None):
        node_keys = list(node.keys())
        n_child = 0
        last_child = None
        for chkey in node_keys:
            if isinstance(node[chkey], dict):
                process(node[chkey], name=chkey)
                n_child += 1
                last_child = chkey
        if n_child == 0:
            # We are a leaf
            return
        if "CHILD" not in node:
            node["CHILD"] = 0.0
        # Accumulate the totals from all children
        for chkey in node_keys:
            if isinstance(node[chkey], dict):
                node["CHILD"] += node[chkey]["TOTAL"]
        # Compute unaccounted time
        if "TOTAL" not in node:
            node["TOTAL"] = node["CHILD"]
        node["OTHER"] = node["TOTAL"] - node["CHILD"]
        if node["OTHER"] < 0:
            # rounding
            node["OTHER"] = 0.0

    process(raw)

    # Restructure the data for plotting

    rows = list()

    def append_counter(path, node):
        node_keys = list(node.keys())
        n_child = 0
        for chkey in node_keys:
            if isinstance(node[chkey], dict):
                append_counter(f"{path}|{chkey}", node[chkey])
                n_child += 1
        if n_child > 0:
            # We have some child counters, so add a counter for untracked
            # time if any.
            if node["OTHER"] > 0.05 * node["TOTAL"]:
                rows.append((f"{path}|Other", node["OTHER"]))
        else:
            # Leaf node, add our value
            rows.append((path, node["TOTAL"]))

    append_counter("main", raw)

    n_column = 0
    for trace, val in rows:
        levels = trace.split("|")
        n_cols = len(levels)
        if n_cols > n_column:
            n_column = n_cols

    lnames = [f"level{x}" for x in range(n_column)]
    data = list()

    for trace, val in rows:
        levels = trace.split("|")
        n_cols = len(levels)
        rlist = list()
        for i in range(n_cols):
            rlist.append(levels[i])
        if n_cols < n_column:
            for i in range(n_cols, n_column):
                rlist.append(None)
        rlist.append(val)
        data.append(rlist)

    cnames = list(lnames)
    cnames.append("value")
    df = pd.DataFrame(data=data, columns=cnames)

    img_px = 800
    pio.kaleido.scope.default_format = "pdf"
    pio.kaleido.scope.default_width = img_px
    pio.kaleido.scope.default_height = img_px
    pio.kaleido.scope.mathjax = None

    fig = px.sunburst(
        data_frame=df,
        path=lnames,
        values="value",
        title="TOAST Timing",
        height=img_px,
        template="plotly",
        branchvalues="total",
    )
    fig.write_image(args.out)


if __name__ == "__main__":
    main()
