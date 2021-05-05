
# Install all the scripts into the bin directory

install(PROGRAMS
    toast_env
    toast_fake_focalplane
    toast_ground_schedule
    toast_satellite_schedule
    toast_benchmark_satellite.py
    DESTINATION bin
)
