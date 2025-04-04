Doc notes:

intro
install
- User install:  conda or pip, include NERSC here
- Dev install:  conda compilers
- Fully custom:  consistent lapack with numpy / scipy

Quickstart:  interactive example, widget

Data model:
    - instrument model:  telescope, site, focalplane
    - detector pointing / response model
    - noise model: mixing
    - Containers
        - Data, Observation, Intervals
    - flagging

Processing model
    - Operator concepts
        - Working with data selections
    - Interactive use
    - Config system

Common Operators
- Pointing model
    - detector pointing
    - pixels: healpix, WCS
    - weights
- Utilities
    - copy / reset / arithmetic
    - flagging intervals
    - memory counting

Data Simulation Operators
- simulated observing
    - schedule ground
    - sim ground
    - schedule satellite
    - sim satellite

- simulated timestream components
    - noise
    - atmosphere
    - sky signal

- simulated instrument effects
    - time constant convolution
    - ground pickup
    - beam asymmetries
    - gain / calibration errors

- 



Data Reduction Operators
- Data quality
    - statistics
    - crosslinking / h_n / etc

- Instrument characterization
    - noise estimation
    - 

- Timestream filtering
    - timeconstant deconvolution
    - common mode
    - 1D poly / 2D poly
    - Ground pickup filter

- Map making
    - Generalized destriper
        - Algorithm, cite madam / npipe
        - Template solver
        - Simple binned map
        - ML limit
    - Specialized filter/bin/obs matrix
    