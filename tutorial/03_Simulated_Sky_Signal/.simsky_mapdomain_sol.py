pysm_sky_cmb = PySMSky(
    comm=None, pixel_indices=None, nside=NSIDE, units="uK_CMB", pysm_sky_config=["c1"]
)

pysm_sky_cmb.exec(local_maps, out="cmb", bandpasses={"ch0": (100, 1), "ch1": (200, 1)})

hp.mollview(local_maps["cmb_ch0"][0] - local_maps["cmb_ch1"][0])
