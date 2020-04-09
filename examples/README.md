# TOAST Example Pipelines

These scripts consist of several example workflows of different types and at different
scales.  The goal of these scripts is to provide both use-case examples and also a
mechanism for benchmarking performance across systems.

## Requirements

You should have an installation of TOAST that matches the version of the examples
directory.  See TOAST installation instructions in the documentation.  Next you should
fetch some input auxiliary data required by the examples:
```bash
./fetch_data.sh
```
**WARNING**:  if you are not running at NERSC, this will download approximately 500MB of
data files.

## Generating the job scripts

To generate all the job directories, do:
```bash
./generate.py
```
This parses the `config.toml` file and creates a job for each system and for every size pipeline test that can run on that system based on memory requirements.  To generate jobs for a new system, you can create a new config stanza for that system and a template job script file to use.

## Running the jobs

For the NERSC jobs, you can go into the job directory and submit the job:
```bash
sbatch run.slurm
```
For local jobs you can do:
```bash
./run.sh
```
