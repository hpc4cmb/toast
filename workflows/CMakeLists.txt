
# Install all the workflows into the bin directory

install(PROGRAMS
    toast_sim_satellite.py
    toast_sim_ground.py
    toast_sim_satellite_simple.py
    toast_sim_ground_simple.py
    DESTINATION bin
)
