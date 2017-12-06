CMake is a requirement for generating the examples.
From this directory, do:

# to download data 
$> ./fetch_data.sh

# to generate slurm files for examples
$> ./generate_slurm.sh

# to generate shell scripts for tiny examples
$> ./generate_shell.sh

To generate job submission scripts for various job sizes.  Cleanup
generated files and job outputs with:

$> ./cleanup.sh

#-----------------------------------------------------------------#

For customization, set the following environment variables:
- TYPES
  - satellite, ground, ground_simple
- SIZES
  - default for SLURM: "large medium small tiny representative"
  - default for SHELL: no customization permitted
- MACHINES
  - default: "cori-intel-knl cori-intel-haswell and edison-intel"
- account
  - default: "mp107"
- queue:
  - default: "debug"

Additionally, any variable in templates/*.in (designated by @VARIABLE@)
or defined in templates/params/* can be overridden by appending
-DVARIABLE="VALUE" to the ./generate_* call.
  - if VALUE has special characters and needs to be quoted in the file (e.g. TIME),
    it is recommended to use form:
      -D<VARIABLE>=\"<VALUE>\"

Example:

  $> ./generate_slurm.sh -DTIME=\"02:00:00\" -DQUEUE="regular" -DACCOUNT="dasrepo"

would change the allocated runtime to 2 hours, the job queue to "regular",
and the account to "dasrepo"

#-----------------------------------------------------------------#

Folder:
- templates/
  - holds the templates for the ground, satellite, and ground_simple
    problem types
- templates/params/
  - set the default parameters for the problem types
- templates/machines/
  - set the default machine parameters for the different machines

#-----------------------------------------------------------------#

For VTune (SLURM scripts are specific to NERSC):

- Must have VTune module loaded to generate SLURM scripts
  - vtune/2018.0 is recommended
  - if not module is loaded, only the shell scripts will be generated
- Run one of the provided scripts to customize the environment variables
  for different collection types:
  
  $> source vtune_params/advanced-hotspots.sh
  $> source vtune_params/concurrency.sh
  $> source vtune_params/general-exploration.sh
  $> source vtune_params/hotspots.sh
  $> source vtune_params/hpc-performance.sh
  $> source vtune_params/locksandwaits.sh
  $> source vtune_params/memory-access.sh
  $> source vtune_params/memory-consumption.sh
  $> source vtune_params/pyhotspots.sh
  $> source vtune_params/tsx-exploration.sh
  $> source vtune_params/tsx-hotspots.sh

