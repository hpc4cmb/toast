
# Install all the scripts into the bin directory

set(ENTRY_POINTS
    toast_env.py
    toast_run.py
    toast_fake_focalplane.py
    toast_fake_telescope.py
    toast_ground_schedule.py
    toast_satellite_schedule.py
    toast_timing_plot.py
    toast_benchmark_satellite.py
    toast_benchmark_ground.py
    toast_benchmark_ground_setup.py
    toast_healpix_convert.py
    toast_healpix_coadd.py
    toast_hdf5_to_spt3g.py
    toast_mini.py
    toast_obsmatrix_combine.py
    toast_obsmatrix_coadd.py
    toast_config_verify.py
    toast_config_compare.py
    toast_merge_timings.py
    toast_plot_wcs.py
    toast_plot_healpix.py
    toast_project_schedule.py
    toast_analyze_schedule.py
    toast_overlap_schedule.py
    toast_map_stats.py
)

foreach(pyscript ${ENTRY_POINTS})
    string(REPLACE ".py" "" outscript "${pyscript}")
    install(PROGRAMS "${pyscript}" DESTINATION bin RENAME "${outscript}")
    install(FILES "${pyscript}" DESTINATION ${PYTHON_SITE}/toast/scripts)
endforeach()

install(FILES
    __init__.py
    benchmarking_utilities.py
    DESTINATION ${PYTHON_SITE}/toast/scripts
)
