
# Benchmarking Performance

TOAST includes several scripts that attempt to mimic a "representative" workflow that simulates and reduces data from either a space or ground-based telescope.  These benchmarks are installed along with the package.  If you have installed a binary package (wheel or conda package), then these may not provide the best absolute performance compared to building from source with the vendor-optimized compilers, etc.  On the other hand, these binaries are built with modern compilers and bundled versions of the latest dependencies built with those compilers.  So the binary packages may actually provide better performance than building from source on older OS flavors.  Before running the benchmarks, install TOAST according to either the [user](install:user) or [development](install:dev) instructions.

## Running the Benchmarks

After installing TOAST, you should have several scripts in you search path, including `toast_benchmark_satellite`, `toast_benchmark_ground`, and `toast_benchmark_ground_setup`.  The satellite benchmark is a good starting point, since it does not require a separate setup step to generate inputs.

### Satellite Telescope Benchmark

Although this script supports many options to tune the parameters of the underlying workflow, the main options for benchmarking are:

```
toast_benchmark_satellite --help
  <snip>
  --out_dir OUT_DIR     The output directory
  --node_mem_gb NODE_MEM_GB
                        Use this much memory per node in GB
  --dry_run DRY_RUN     Comma-separated total_procs,node_procs to simulate.
  --case {auto,tiny,xsmall,small,medium,large,xlarge,heroic}
                        Size of the worflow to be run: 'tiny' (1GB), 'xsmall' (10GB),
                        'small' (100GB), 'medium' (1TB), 'large' (10TB),
                        'xlarge' (100TB), 'heroic' (1000TB) or
                        'auto' (deduced from system parameters).

  (more options to the underlying data operations follow...)
```

The "case" can be used to select a fixed-size data volume.  If "auto" is specified, it will attempt to use as much data as possible to fill the available memory.  You can set the `--node_mem_gb` to something smaller than the actual memory in order to adjust the data volume.  Before launching real jobs, it is useful to test the job setup in dry-run mode.  You can
do these tests serially.

#### Dry Run

The `--dry_run` option can be used to try different concurrencies and data volumes.  For example, to simulate the
job setup for running 1024 MPI processes with 16 processes per node (so 64 nodes), on a
system with 90GB of RAM per node, you can do:

```{code-block} bash
toast_benchmark_satellite --node_mem_gb 90 --case auto --dry_run 1024,16

TOAST INFO: TOAST version = 2.3.12.dev308
TOAST INFO: Using a maximum of 4 threads per process
TOAST INFO: Running with 1 processes at 2022-04-05 11:26:00.503491
TOAST INFO: DRY RUN simulating 1024 total processes with 16 per node
TOAST INFO: Minimum detected per-node memory available is 118.01 GB
TOAST INFO: Setting per-node available memory to 90.00 GB as requested
TOAST INFO: Job has 64 total nodes
TOAST WARNING: Mission start time '2022-04-05 11:26:00.535054' is not timezone-aware.  Assuming UTC.
TOAST INFO: Using automatic workflow size selection (case='auto') with 1.00 GB reserved for per process overhead.
TOAST INFO: Distribution using:
  2054 detectors and 190 observations
  140493209740 total samples
  64 groups of 1 nodes with 16 processes each
  5750.90 GB predicted memory use   (5760.00 GB available)
  ('auto' workflow size)
TOAST INFO: Using 1027 hexagon-packed pixels.
TOAST INFO: Exit from dry run.
```

When running the script without the `--dry_run` option, there will be several input files created

#### Starting Small

A good starting point is to begin with a single-node job.  Choose how many processes you
will be using per node.  Things to consider:

1.  Most of the parallelism in TOAST is process-level using MPI.  There is some limited use of OpenMP, but running with more MPI ranks generally leads to better performance at the moment.

2.  There is some additional memory overhead on each node, so running nodes "fully packed" with MPI processes may not be possible.  You should experiment with different numbers of MPI ranks per node.

3.  Make sure to set `OMP_NUM_THREADS` appropriately so that the `(MPI ranks per node) X (OpenMP threads)` equals the total number of physical cores on each node.  Using more threads than physical cores (hyperthreading) generally produces worse results.

Here is an example running interactively on `cori.nersc.gov` (KNL nodes), which uses the SLURM
scheduler:

```{code-block} bash
# 16 ranks per node, each with 4 threads.
# 4 cores left for OS use.
# Depth is 16, due to 4 hyperthreads per core.
%>  export OMP_NUM_THREADS=4
%>  srun -C knl -q interactive -t 00:30:00 -N 1 -n 16 -c 16 \
toast_benchmark_satellite
```

#### Scaling Up

After doing dry-run tests and running very small jobs you can increase the node count to
support something like the small / medium workflow cases.  At this point you can test
the effects of adjusting the number of MPI processes per node.  After you have found a
configuration that seems the best, increase the node count again to run the larger
cases.

### Ground-based Telescope Benchmark

The benchmark for a ground-based telescope is slightly more complicated, since it requires running a separate script to setup job inputs.  This "setup" script constructs inputs for all possible cases of data volumes, even if only a tiny job will be run.  The setup script plans a year's worth of observing and simulates fake atmosphere for each observation.  The setup script will generate 1-2TB of this input data to accomodate the largest benchmark jobs.

```{admonition} To-Do
Should we support / document generating less that this large set of input files?
```

The setup script uses MPI in an embarrassingly parallel way and benefits from as many processes as you can reasonably allocate to the job.  As another example at NERSC (using the haswell nodes):

```{code-block} bash
# Fully packed MPI processes.
%>  export OMP_NUM_THREADS=1
%>  srun -C haswell -q interactive -t 04:00:00 -N 4 -n 128 -c 2 \
toast_benchmark_ground_setup
```

If your job runs out of time, just re-execute it and it will continue where it left off.

After running this setup, you can use the `toast_benchmark_ground` script in a similar fashion to the satellite benchmark above.  One difference is that this script expects a directory of inputs (which defaults to the default name used in the setup script):

```
toast_benchmark_ground --help
  <snip>
  --input_dir INPUT_DIR
                        The input directory
  --out_dir OUT_DIR     The output directory
  --node_mem_gb NODE_MEM_GB
                        Use this much memory per node in GB
  --dry_run DRY_RUN     Comma-separated total_procs,node_procs to simulate.
  --case {auto,tiny,xsmall,small,medium,large,xlarge,heroic}
                        Size of the worflow to be run: 'tiny' (1GB), 'xsmall' (10GB),
                        'small' (100GB), 'medium' (1TB), 'large' (10TB),
                        'xlarge' (100TB), 'heroic' (1000TB) or
                        'auto' (deduced from system parameters).

  (more options to the underlying data operations follow...)
```


## Metrics

At the end of the job a "Science Metric" is reported.  This is just the total number of timestream samples (across all detectors) divided by the total node-seconds.  It gives an approximate relative measure of how efficiently the job ran for a particular data volume.  Note that this metric will be biased towards fewer large nodes.  To make it more meaningful, one should include the power use per node to get the "Data processed per Watt".

```{important}
Changing other parameters for the simulation and reduction operations used can have dramatic impacts on the code performance.  When comparing performance across jobs, only the data volume (number of samples) should be changed.
```
