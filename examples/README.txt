From this directory, do:

$> ./generate.sh

To generate job submission scripts for various job sizes.  Cleanup
generated files and job outputs with:

$> ./cleanup.sh

#-----------------------------------------------------------------#

For VTune:

- Some modifications were made to account for different runtimes,
  the need to submit jobs to regular queue instead of debug,
  and increasing the runtime, etc.
- It uses the CMake "configure_file" utility, so you can either
  load a specific CMake module or it will do that for you

Folder:
- templates/
  - holds the templates for the ground, satellite, and ground_simple
    problem types
- templates/params/
  - set the default parameters for the problem types
- templates/machines/
  - set the default machine parameters for the different machines

Must have VTune module loaded, vtune/2018.0 is recommended

- To override the default settings, such as "ACCOUNT" (job submission repo),
  "QUEUE", "TIME" (max runtime), etc. then define them as arguments to
  ./generate_vtune.sh in the format:
      -D<VARIABLE>=<VALUE>
  if VALUE has special characters and needs to be quoted in the file (e.g. TIME),
  it is recommended to use form:
      -D<VARIABLE>=\"<VALUE>\"

