
# Install all the scripts into the bin directory

install(PROGRAMS
    toast_env_test.py
    toast_cache_test.py
    toast_cov_invert.py
    toast_cov_rcond.py
    toast_fake_focalplane.py
    toast_ground_schedule.py
    toast_satellite_sim.py
    toast_ground_sim.py
    toast_ground_sim_simple.py
    toast_benchmark.py
    DESTINATION bin
)
