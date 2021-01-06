#!/usr/bin/env python3

import os
import sys
import re

import csv

import numpy as np

import healpy as hp

import matplotlib

matplotlib.use("pdf")

import matplotlib.pyplot as plt


# Load all the cases

cases = dict()

tmpat = re.compile(r"mapcomm_nproc-(.*)_nside-(.*)_nsub-(.*)\.csv")
mappat = re.compile(
    r"mapcomm_nproc-(.*)_nside-(.*)_nsub-(.*)_full-(.*)_empty-(.*)_fill-(.*)_cover\.fits"
)
rootmappat = re.compile(
    r"mapcomm_nproc-(.*)_nside-(.*)_nsub-(.*)_full-(.*)_empty-(.*)_fill-(.*)_cover-root\.fits"
)

allreduce_pat = re.compile(r"SYNC_ALLREDUCE_(\d+)_(\d+)_(\d+)")

alltoallv_pat = re.compile(r"SYNC_ALLTOALLV_(\d+)_(\d+)_(\d+)")


def case_init(tcases, nside, full, fill):
    if full not in cases:
        tcases[full] = dict()
    if fill not in cases[full]:
        tcases[full][fill] = dict()
    if nside not in cases[full][fill]:
        tcases[full][fill][nside] = dict()
        tcases[full][fill][nside]["allreduce"] = dict()
        tcases[full][fill][nside]["alltoallv"] = dict()
    return


maxproc = 0
maxproc_root = 0

for dirpath, dirnames, filenames in os.walk("."):
    for file in filenames:
        path = os.path.join(dirpath, file)

        mat = mappat.match(file)
        if mat is not None:
            print("found map file {}".format(path))
            nproc = int(mat.group(1))
            nside = int(mat.group(2))
            nsub = int(mat.group(3))
            full = int(mat.group(4))
            empty = int(mat.group(5))
            fill = int(mat.group(6))
            case_init(cases, nside, full, fill)
            if nproc >= maxproc:
                cases[full][fill][nside]["cover"] = path
                maxproc = nproc

        mat = rootmappat.match(file)
        if mat is not None:
            print("found map root file {}".format(path))
            nproc = int(mat.group(1))
            nside = int(mat.group(2))
            nsub = int(mat.group(3))
            full = int(mat.group(4))
            empty = int(mat.group(5))
            fill = int(mat.group(6))
            case_init(cases, nside, full, fill)
            if nproc >= maxproc_root:
                cases[full][fill][nside]["rootcover"] = path
                maxproc_root = nproc

        mat = tmpat.match(file)
        if mat is not None:
            print("found timing file {}".format(path))
            nproc = int(mat.group(1))
            nside = int(mat.group(2))
            nsub = int(mat.group(3))
            with open(path, "r") as tf:
                reader = csv.reader(tf, delimiter=",")
                for row in reader:
                    namemat = allreduce_pat.match(row[0])
                    if namemat is not None:
                        print("Found allreduce timing {}".format(row[0]))
                        full = int(namemat.group(1))
                        empty = int(namemat.group(2))
                        fill = int(namemat.group(3))
                        case_init(cases, nside, full, fill)
                        seconds = float(row[7]) / 5.0
                        cases[full][fill][nside]["allreduce"][nproc] = seconds
                    namemat = alltoallv_pat.match(row[0])
                    if namemat is not None:
                        print("Found alltoallv timing {}".format(row[0]))
                        full = int(namemat.group(1))
                        empty = int(namemat.group(2))
                        fill = int(namemat.group(3))
                        case_init(cases, nside, full, fill)
                        seconds = float(row[7]) / 5.0
                        cases[full][fill][nside]["alltoallv"][nproc] = seconds

print(cases)

n_case = 0
fullvals = sorted(cases.keys())
for full in fullvals:
    fillvals = sorted(cases[full].keys())
    n_case += len(fillvals)

fig = plt.figure(figsize=(12, 2.5 * n_case), dpi=100)

plotrows = 0
for x in cases.keys():
    for y in cases[x].keys():
        plotrows += 1

plotoff = 1

fullvals = sorted(cases.keys())
for full in fullvals:
    fillvals = sorted(cases[full].keys())
    for fill in fillvals:
        # Find the lowest NSIDE to use for plotting, since these submap coverage plots
        # will be identical for every NSIDE.
        nsides = sorted(cases[full][fill].keys())
        pmax = 0
        for ns in nsides:
            procs = sorted(cases[full][fill][ns]["allreduce"].keys())
            for p in procs:
                if p > pmax:
                    pmax = p

        cover = hp.read_map(cases[full][fill][nsides[0]]["cover"])
        rootcover = hp.read_map(cases[full][fill][nsides[0]]["rootcover"])
        hp.mollview(
            map=cover,
            sub=(plotrows, 3, plotoff),
            title="Total Submap Coverage {:d}% / {:d}%".format(full, fill),
            xsize=1200,
            cmap="rainbow",
            min=0,
            max=pmax,
            margins=(0.0, 0.0, 0.0, 0.01),
        )
        plotoff += 1
        hp.mollview(
            map=rootcover,
            sub=(plotrows, 3, plotoff),
            title="Rank Zero Submaps {:d}% / {:d}%".format(full, fill),
            xsize=1200,
            cmap="rainbow",
            min=0,
            max=1,
            margins=(0.0, 0.0, 0.0, 0.01),
        )
        plotoff += 1
        ax = fig.add_subplot(plotrows, 3, plotoff)
        for ns in nsides:
            lw = 0.5 * (ns // 256)
            procs = sorted(cases[full][fill][ns]["allreduce"].keys())
            xdata = np.array(procs, dtype=np.int32)
            ydata = np.array([cases[full][fill][ns]["allreduce"][x] for x in procs])
            ax.plot(
                xdata,
                ydata,
                label="allreduce N{}".format(ns),
                color="r",
                linewidth=lw,
                marker="o",
                markersize=(lw + 1),
            )
            ydata = np.array([cases[full][fill][ns]["alltoallv"][x] for x in procs])
            ax.plot(
                xdata,
                ydata,
                label="alltoallv N{}".format(ns),
                color="g",
                linewidth=lw,
                marker="o",
                markersize=(lw + 1),
            )
        ax.legend(loc="upper left", fontsize=6)

        ax.set_ylabel("Seconds (Mean of {} calls)".format(5))
        ax.set_xlabel("MPI Ranks")
        ax.set_ylim(0, 9.0)
        plotoff += 1

plt.tight_layout()
# plt.subplots_adjust(top=0.9)
pfile = "mapcomm.pdf"
plt.savefig(pfile)
plt.close()
