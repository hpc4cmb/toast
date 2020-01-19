"""Helper tools common to all notebooks.
"""

import os
import numpy as np
import re

from toast.tod import hex_pol_angles_radial, hex_pol_angles_qu, hex_layout


def check_nersc(reservation=None, repo=None):
    """Check if we are running at NERSC.
    
    If we are at NERSC, select the account to use for batch jobs.
    
    Args:
        reservation (str):  Attempt to use this reservation for slurm jobs.
        repo (str):  In the case of multiple repos, use this one.
        
    Returns:
        (tuple):  The (host, repo, reservation) configured.

    """
    nersc_host = None
    nersc_repo = None
    nersc_resv = None
    if "NERSC_HOST" in os.environ:
        # We are at NERSC
        nersc_host = os.environ["NERSC_HOST"]
        # Be kind to the login node
        os.environ["OMP_NUM_THREADS"] = "4"
        import subprocess as sp
        repos = sp.check_output(
            "getnim -U $(whoami) | awk '{print $1}'", 
            shell=True,
            universal_newlines=True
        ).split()
        print(
            "Running on NERSC machine '{}'\n  with access to repos: {}".format(
                nersc_host, ", ".join(repos)
            )
        )
        if repo is not None:
            if repo in repos:
                nersc_repo = repo
                print("Using requested repo {}".format(repo))
            else:
                print("Requested repo {} not in list of enabled repos".format(repo))
        if nersc_repo is None:
            nersc_repo = repos[0]
            print("Using default repo {}".format(nersc_repo))
        if reservation is not None:
            # We would like to use a reservation
            checkres = sp.check_output(
                "scontrol show reservation {}".format(reservation), 
                shell=True,
                universal_newlines=True
            ).split()
            # Does this reservation even exist?
            if re.match(r".*not found.*", checkres[0]) is not None:
                print(
                    "Reservation '{}' does not exist or is expired".format(reservation)
                )
            else:
                startiso = None
                stopiso = None
                fullres = " ".join(checkres)
                startmat = re.match(r".*StartTime=(\S*).*", fullres)
                stopmat = re.match(r".*EndTime=(\S*).*", fullres)
                if startmat is not None:
                    startiso = startmat.group(1)
                if stopmat is not None:
                    stopiso = stopmat.group(1)
                if (startiso is None) or (stopiso is None):
                    print(
                        "Could not parse scontrol output for reservation '{}'"
                        .format(reservation)
                    )
                else:
                    from datetime import datetime
                    start = datetime.strptime(startiso, "%Y-%m-%dT%H:%M:%S")
                    stop = datetime.strptime(stopiso, "%Y-%m-%dT%H:%M:%S")
                    now = datetime.now()
                    print(
                        "Reservation '{}' valid from {} to {}".format(
                            reservation,
                            start.isoformat(),
                            stop.isoformat()
                        )
                    )
                    print("Current time is {}".format(now.isoformat()))
                    if (now >= start) and (now < stop):
                        print("Selecting reservation '{}'".format(reservation))
                        nersc_resv = reservation
                    else:
                        print("Reservation '{}' not currently valid".format(reservation))
    else:
        print("Not running at NERSC, slurm jobs disabled.")
    return (nersc_host, nersc_repo, nersc_resv)


def fake_focalplane(
    samplerate=20,
    epsilon=0,
    net=1,
    fmin=0,
    alpha=1,
    fknee=0.05,
    fwhm=30,
    npix=7,
    fov=3.0
):
    """Create a set of fake detectors.

    This generates 7 pixels (14 dets) in a hexagon layout at the boresight
    and with a made up polarization orientation.

    Args:
        None

    Returns:
        (dict):  dictionary of detectors and their properties.

    """
    zaxis = np.array([0, 0, 1.0])
    
    pol_A = hex_pol_angles_qu(npix)
    pol_B = hex_pol_angles_qu(npix, offset=90.0)
    
    dets_A = hex_layout(npix, fov, "", "", pol_A)
    dets_B = hex_layout(npix, fov, "", "", pol_B)
    
    dets = dict()
    for p in range(npix):
        pstr = "{:01d}".format(p)
        for d, layout in zip(["A", "B"], [dets_A, dets_B]):
            props = dict()
            props["quat"] = layout[pstr]["quat"]
            props["epsilon"] = epsilon
            props["rate"] = samplerate
            props["alpha"] = alpha
            props["NET"] = net
            props["fmin"] = fmin
            props["fknee"] = fknee
            props["fwhm_arcmin"] = fwhm
            dname = "{}{}".format(pstr, d)
            dets[dname] = props
    return dets
