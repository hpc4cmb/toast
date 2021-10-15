
# Install all the scripts into the bin directory

install(PROGRAMS
    toast_env
    toast_fake_focalplane
    toast_ground_schedule
    toast_satellite_schedule
    toast_timing_plot
    toast_benchmark_satellite
    toast_benchmark_ground
    toast_benchmark_ground_setup
    benchmarking_utilities.py
    toast_healpix_convert
    DESTINATION bin
)

install(FILES
    __init__.py
    benchmarking_utilities.py
    DESTINATION ${PYTHON_SITE}/toast/scripts
)
