"""Tools for use in the documentation notebooks.
"""
import os
import sys
import tempfile
import shutil

import numpy as np
from astropy import units as u
import astropy.io.fits as af
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

import toast
import toast.ops

from toast.instrument import (
    Telescope,
    Focalplane,
    GroundSite,
    SpaceSite,
)
from toast.instrument_sim import fake_hexagon_focalplane

from toast.schedule import GroundSchedule
from toast.schedule_sim_ground import run_scheduler
from toast.schedule_sim_satellite import create_satellite_schedule

from toast.observation import default_values as defaults


def create_outdir(top_dir, sub_dir=None, comm=None):
    """Create a directory for notebook output.

    Args:
        top_dir (str): path to the top level output directory, relative to pwd.
        sub_dir (str): sub directory to create.
        comm (MPI.Comm): the MPI communicator (or None).

    Returns:
        str: full path to the sub_dir if specified, else the top_dir.

    """
    pwd = os.path.abspath(".")
    top = os.path.join(pwd, top_dir)
    retdir = top
    if sub_dir is not None:
        retdir = os.path.join(top, sub_dir)
    if (comm is None) or (comm.rank == 0):
        if not os.path.isdir(top):
            os.mkdir(top)
        if not os.path.isdir(retdir):
            os.mkdir(retdir)
    if comm is not None:
        comm.barrier()
    return retdir


def create_space_telescope(group_size, sample_rate=10.0 * u.Hz, pixels_per_process=1):
    """Create a fake satellite telescope with at least one pixel per process."""
    npix = 1
    ring = 1
    while 2 * npix <= group_size * pixels_per_process:
        npix += 6 * ring
        ring += 1
    fp = fake_hexagon_focalplane(
        n_pix=npix,
        sample_rate=sample_rate,
        psd_fmin=1.0e-5 * u.Hz,
        psd_net=0.05 * u.K * np.sqrt(1 * u.second),
        psd_fknee=(sample_rate / 2000.0),
    )
    site = SpaceSite("L2")
    return Telescope("test", focalplane=fp, site=site)


def create_ground_telescope(
    group_size, sample_rate=10.0 * u.Hz, pixels_per_process=1, fknee=None
):
    """Create a fake ground telescope with at least one detector per process."""
    npix = 1
    ring = 1
    while 2 * npix <= group_size * pixels_per_process:
        npix += 6 * ring
        ring += 1
    if fknee is None:
        fknee = sample_rate / 2000.0
    fp = fake_hexagon_focalplane(
        n_pix=npix,
        sample_rate=sample_rate,
        psd_fmin=1.0e-5 * u.Hz,
        psd_net=0.05 * u.K * np.sqrt(1 * u.second),
        psd_fknee=fknee,
    )

    site = GroundSite("Atacama", "-22:57:30", "-67:47:10", 5200.0 * u.meter)
    return Telescope("telescope", focalplane=fp, site=site)


def fake_satellite_observing(
    mpicomm,
    obs_per_group=1,
    sample_rate=10.0 * u.Hz,
    obs_time=10.0 * u.minute,
    pixels_per_process=1,
    hwp_rpm=10.0,
):
    """Create a data object with a simple satellite sim.

    Use the specified MPI communicator to attempt to create 2 process groups.  Create
    a fake telescope and run the satellite sim to make some observations for each
    group.  This is useful for testing many operators that need some pre-existing
    observations with boresight pointing.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        obs_per_group (int): the number of observations assigned to each group.
        sample_rate (Quantity): the sample rate.
        obs_time (Quantity): the time length of one observation.

    Returns:
        toast.Data: the distributed data with named observations.

    """
    toastcomm = toast.Comm(world=mpicomm)
    data = toast.Data(toastcomm)

    tele = create_space_telescope(
        toastcomm.group_size,
        sample_rate=sample_rate,
        pixels_per_process=pixels_per_process,
    )

    # Create a schedule

    sch = create_satellite_schedule(
        prefix="test_",
        mission_start=datetime(2023, 2, 23),
        observation_time=obs_time,
        gap_time=0 * u.minute,
        num_observations=(toastcomm.ngroups * obs_per_group),
        prec_period=10 * u.minute,
        spin_period=1 * u.minute,
    )

    # Scan fast enough to cover some sky in a short amount of time.  Reduce the
    # angles to achieve a more compact hit map.
    if hwp_rpm == 0 or hwp_rpm is None:
        hwp_angle = None
    else:
        hwp_angle = defaults.hwp_angle
    sim_sat = toast.ops.SimSatellite(
        name="sim_sat",
        telescope=tele,
        schedule=sch,
        hwp_angle=hwp_angle,
        hwp_rpm=hwp_rpm,
        spin_angle=5.0 * u.degree,
        prec_angle=10.0 * u.degree,
        detset_key="pixel",
    )
    sim_sat.apply(data)
    return data


def fake_ground_observing_smallpatch(
    mpicomm=None,
    sample_rate=50.0 * u.Hz,
    temp_dir=None,
    el_nod=False,
    el_nods=[-1 * u.degree, 1 * u.degree],
    pixels_per_process=1,
    fknee=None,
):
    """Create a data object with a simple ground sim.

    Args:
        mpicomm (MPI.Comm): the MPI communicator (or None).
        sample_rate (Quantity): the sample rate.

    Returns:
        toast.Data: the distributed data with named observations.

    """
    toastcomm = toast.Comm(world=mpicomm)
    data = toast.Data(toastcomm)

    tele = create_ground_telescope(
        toastcomm.group_size,
        sample_rate=sample_rate,
        pixels_per_process=pixels_per_process,
        fknee=fknee,
    )

    # Create a schedule.

    schedule = None

    if mpicomm is None or mpicomm.rank == 0:
        tdir = temp_dir
        if tdir is None:
            tdir = tempfile.mkdtemp()

        sch_file = os.path.join(tdir, "ground_schedule.txt")
        run_scheduler(
            opts=[
                "--site-name",
                tele.site.name,
                "--telescope",
                tele.name,
                "--site-lon",
                "{}".format(tele.site.earthloc.lon.to_value(u.degree)),
                "--site-lat",
                "{}".format(tele.site.earthloc.lat.to_value(u.degree)),
                "--site-alt",
                "{}".format(tele.site.earthloc.height.to_value(u.meter)),
                "--patch",
                "small_patch,1,40,-40,44,-44",
                "--start",
                "2025-01-01 00:00:00",
                "--stop",
                "2025-01-01 06:00:00",
                "--out",
                sch_file,
            ]
        )
        schedule = GroundSchedule()
        schedule.read(sch_file)
        if temp_dir is None:
            shutil.rmtree(tdir)
    if mpicomm is not None:
        schedule = mpicomm.bcast(schedule, root=0)

    sim_ground = toast.ops.SimGround(
        name="sim_ground",
        telescope=tele,
        schedule=schedule,
        hwp_angle=defaults.hwp_angle,
        hwp_rpm=120.0,
        weather="atacama",
        median_weather=True,
        detset_key="pixel",
        elnod_start=el_nod,
        elnods=el_nods,
        scan_accel_az=3 * u.degree / u.second**2,
    )
    sim_ground.apply(data)

    return data


def fake_input_map_wcs(data, dist_key, out_key, pixop, fig=None):
    """Make a completely fake sky map to scan.

    This uses the WCS information from the pixel operator and the precomputed
    pixel distribution.

    """
    dist = data[dist_key]
    pix_data = toast.pixels.PixelData(dist, np.float64, n_value=3)
    
    input_data = None
    
    if data.comm.world_rank == 0:
        # Create the fake map data on one process
        wcs = pixop.wcs
        shape = pixop.wcs_shape

        lon_dim = shape[0]
        lat_dim = shape[1]

        np.random.seed(987654321)
        image = np.random.normal(loc=0.0, scale=100.0, size=(lat_dim, lon_dim)) 
        image = gaussian_filter(image, sigma=(lat_dim/40, lon_dim/40))

        # We are going to artificially create Q/U signals that are just scaled
        # versions of the intensity.
        input_data = np.zeros((3, lat_dim * lon_dim), dtype=np.float64)
        input_data[0] = np.transpose(image).flatten()
        scale = 0.001
        input_data[1] = scale * input_data[0]
        input_data[2] = -scale * input_data[0]
        imin = np.amin(image)
        imax = np.amax(image)
        qmin = scale * imin
        qmax = scale * imax
        umin = -qmax
        umax = -qmin

        if fig is not None:
            ax = fig.add_subplot(1, 3, 1)
            x, y = np.meshgrid(np.arange(lon_dim), np.arange(lat_dim))
            cs = ax.contourf(x, y, image, 100, cmap='jet')
            cbar = plt.colorbar(cs, orientation="horizontal")
            cbar.set_ticks([imin, imax])
            cbar.set_ticklabels([f"{imin:0.1e}", f"{imax:0.1e}"])
            ax.set_title("Stokes I")

            ax = fig.add_subplot(1, 3, 2)
            x, y = np.meshgrid(np.arange(lon_dim), np.arange(lat_dim))
            cs = ax.contourf(x, y, scale * image, 100, cmap='jet')
            cbar = plt.colorbar(cs, orientation="horizontal")
            cbar.set_ticks([qmin, qmax])
            cbar.set_ticklabels([f"{qmin:0.1e}", f"{qmax:0.1e}"])
            ax.set_title("Stokes Q")

            ax = fig.add_subplot(1, 3, 3)
            x, y = np.meshgrid(np.arange(lon_dim), np.arange(lat_dim))
            cs = ax.contourf(x, y, -scale * image, 100, cmap='jet')
            cbar = plt.colorbar(cs, orientation="horizontal")
            cbar.set_ticks([umin, umax])
            cbar.set_ticklabels([f"{umin:0.1e}", f"{umax:0.1e}"])
            ax.set_title("Stokes U")
    
    # Distribute the map to all processes
    pix_data.broadcast_map(input_data)
    data[out_key] = pix_data


def simulate_ground_data(data, fig=None):
    """Simulate simple fake detector signal.

    This function separately simulates sky, instrument noise, and atmosphere
    and also co-adds them to make the final signal.

    """
    # Detector quaternion pointing in Az/El
    det_pointing_azel = toast.ops.PointingDetectorSimple(
        boresight="boresight_azel", quats="quats_azel", shared_flags=None
    )
    # Detector quaternion pointing in RA/DEC
    det_pointing_radec = toast.ops.PointingDetectorSimple(
        boresight="boresight_radec", quats="quats_radec", shared_flags=None
    )

    # Detector Stokes weights (I/Q/U)
    weights_radec = toast.ops.StokesWeights(weights="weights_radec", mode="IQU")
    weights_radec.detector_pointing = det_pointing_radec

    weights_azel = toast.ops.StokesWeights(weights="weights_azel", mode="IQU")
    weights_azel.detector_pointing = det_pointing_azel

    # Simple RA/DEC pixelization:  CAR projection and Stokes I/Q/U weights.
    pixels_radec = toast.ops.PixelsWCS(
        detector_pointing=det_pointing_radec,
        projection="CAR",
        resolution=(0.01 * u.degree, 0.01 * u.degree),
        auto_bounds=True,
    )

    # Build the pixel distribution
    build_dist = toast.ops.BuildPixelDistribution(
        pixel_pointing=pixels_radec
    )
    build_dist.apply(data)

    # Create default noise model
    default_model = toast.ops.DefaultNoiseModel()
    default_model.apply(data)

    # Create elevation-weighted noise model
    el_weighted_model = toast.ops.ElevationNoise(
        noise_model=default_model.noise_model,
        out_model="el_noise_model",
        detector_pointing=det_pointing_azel
    )
    el_weighted_model.apply(data)

    # Pre-create the timestreams for each component separately, rather
    # than accumulating to the final signal.
    for ob in data.obs:
        ob.detdata.create("sky")
        ob.detdata.create("noise")
        ob.detdata.create("atmosphere")
        ob.detdata.create("DC")

    # Create a fake DC level for each detector
    DC_scale = 5.0
    for ob in data.obs:
        dc = DC_scale * (np.random.random_sample(size=len(ob.local_detectors)) - 0.5)
        for idet, det in enumerate(ob.local_detectors):
            ob.detdata["DC"][det, :] = dc[idet]

    # Create a fake input sky
    fake_input_map_wcs(data, build_dist.pixel_dist, "input", pixels_radec, fig=fig)

    # Simulate fake sky timestreams from the fake map.  To prevent
    # storing the detector pointing, we wrap this in a pipeline over
    # individual detectors.

    toast.ops.Reset(detdata=["sky"]).apply(data)

    scan_map = toast.ops.ScanMap(
        map_key="input",
        pixels=pixels_radec.pixels,
        weights=weights_radec.weights,
        det_data="sky",
    )

    scan_pipe = toast.ops.Pipeline(
        operators=[
            pixels_radec,
            weights_radec,
            scan_map,
        ],
        detector_sets=["SINGLE"],
    )
    scan_pipe.apply(data)

    # Simulate detector instrument noise from the elevation-weighted noise model

    toast.ops.Reset(detdata=["noise"]).apply(data)
    sim_noise = toast.ops.SimNoise(
        noise_model=el_weighted_model.out_model,
        det_data="noise",
    )
    sim_noise.apply(data)

    # Simulate atmosphere signal.  Here we simulate two components:  One coarse component
    # with larger structures and one finer one.

    toast.ops.Reset(detdata=["atmosphere"]).apply(data)

    sim_atm = toast.ops.SimAtmosphere(
        detector_pointing=det_pointing_azel,
        det_data="atmosphere",
        shared_flags=None,
        xstep=10 * u.meter,
        ystep=10 * u.meter,
        zstep=10 * u.meter,
        lmin_center=0.001 * u.meter,
        lmin_sigma=0.0001 * u.meter,
        lmax_center=1.0 * u.meter,
        lmax_sigma=0.1 * u.meter,
        gain=6.0e-5,
        realization=0,
        zatm=40000 * u.meter,
        zmax=200 * u.meter,
        nelem_sim_max=10000,
        wind_dist=3000.0 * u.meter,
        z0_center=2000.0 * u.meter,
    )
    sim_atm.apply(data)

    sim_atm_coarse = toast.ops.SimAtmosphere(
        detector_pointing=det_pointing_azel,
        det_data="atmosphere",
        add_loading=False,
        shared_flags=None,
        lmin_center=300 * u.meter,
        lmin_sigma=30 * u.meter,
        lmax_center=10000 * u.m,
        lmax_sigma=1000 * u.m,
        xstep=100 * u.m,
        ystep=100 * u.m,
        zstep=100 * u.m,
        zmax=2000 * u.m,
        nelem_sim_max=30000,
        gain=6e-4,
        realization=1000000,
        wind_dist=10000 * u.m,
    )
    sim_atm_coarse.apply(data)

    # Combine all these into the simulated "signal"
    toast.ops.Reset(detdata=["signal"]).apply(data)
    toast.ops.Combine(first="sky", second="signal", result="signal", op="add").apply(data)
    toast.ops.Combine(first="noise", second="signal", result="signal", op="add").apply(data)
    toast.ops.Combine(first="atmosphere", second="signal", result="signal", op="add").apply(data)
    toast.ops.Combine(first="DC", second="signal", result="signal", op="add").apply(data)

    # Delete data objects used in the simulation.
    del data[build_dist.pixel_dist]
    toast.ops.Delete(
        meta=[default_model.noise_model, el_weighted_model.out_model],
        detdata=[
            pixels_radec.pixels,
            weights_azel.weights,
            weights_radec.weights,
            det_pointing_azel.quats,
            det_pointing_radec.quats,
        ]
    ).apply(data)


def plot_wcs_maps(root):
    figsize = (12, 6)
    figdpi = 100

    hit_file = f"{root}_hits.fits"
    map_file = f"{root}_map.fits"

    def plot_single(wcs, hdata, hindx, vmin, vmax, title):
        fig = plt.figure(figsize=figsize, dpi=figdpi)
        ax = fig.add_subplot(projection=wcs, slices=("x", "y", hindx))
        im = ax.imshow(np.transpose(hdu.data[hindx, :, :]), cmap="jet", vmin=vmin, vmax=vmax)
        ax.grid(color="white", ls="solid")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(im, orientation="vertical")
        plt.suptitle(title)
        plt.show()

    def map_range(hdata):
        minval = np.amin(hdata)
        maxval = np.amax(hdata)
        margin = 0.1 * (maxval - minval)
        if margin == 0:
            margin = -1
        minval += margin
        maxval -= margin
        return minval, maxval
    
    def sym_range(hdata):
        minval, maxval = map_range(hdata)
        ext = max(np.absolute(minval), np.absolute(maxval))
        return -ext, ext

    def sub_mono(hitdata, mdata):
        if hitdata is None:
            return
        goodpix = np.logical_and(
            (hitdata > 0),
            (mdata != 0)
        )
        mono = np.mean(mdata[goodpix])
        print(f"Monopole = {mono}")
        mdata[goodpix] -= mono
        mdata[np.logical_not(goodpix)] = 0

    hdulist = af.open(hit_file)
    hdu = hdulist[0]
    hitdata = np.array(hdu.data[0, :, :])
    wcs = WCS(hdu.header)
    maxhits = np.amax(hdu.data[0, :, :])
    plot_single(wcs, hdu, 0, 0, maxhits, "Hit Counts")
    del hdu
    hdulist.close()

    hdulist = af.open(map_file)
    hdu = hdulist[0]
    wcs = WCS(hdu.header)
    sub_mono(hitdata, hdu.data[0, :, :])
    mmin, mmax = sym_range(hdu.data[0, :, :])
    plot_single(wcs, hdu, 0, mmin, mmax, "Stokes I")
    mmin, mmax = sym_range(hdu.data[1, :, :])
    plot_single(wcs, hdu, 1, mmin, mmax, "Stokes Q")
    mmin, mmax = sym_range(hdu.data[2, :, :])
    plot_single(wcs, hdu, 2, mmin, mmax, "Stokes U")
    del hdu
    hdulist.close()

    del hitdata
