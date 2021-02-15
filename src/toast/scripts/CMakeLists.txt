
# Install all the scripts into the bin directory

install(PROGRAMS
    toast_env
    toast_cov_invert
    toast_cov_rcond
    toast_fake_focalplane
    toast_ground_schedule
    toast_satellite_schedule
    toast_benchmark
    DESTINATION bin
)
