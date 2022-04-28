# Change Log

## 2.3.13 (Unreleased)

-   Nothing yet.

## 2.3.12 (2020-11-12)

-   Fix typo in MKL cmake checks which broke OS X builds with MKL.
-   Fix typo in toast\_ground\_schedule.py pipeline.

## 2.3.11 (2020-11-12)

-   Restore support for directly using Intel MKL for FFTs (PR
    [\#371](https://github.com/hpc4cmb/toast/pull/371),
    [\#372](https://github.com/hpc4cmb/toast/pull/372)).
-   Fixed bug in wind velocity during atmosphere simulations.

## 2.3.10 (2020-10-15)

-   Run serial unit tests without MPI. Fix bug in ground filter (PR
    [\#370](https://github.com/hpc4cmb/toast/pull/370)).

## 2.3.9 (2020-10-15)

-   Add stand-alone benchmarking tool (PR
    [\#365](https://github.com/hpc4cmb/toast/pull/365)).
-   Update wheels to use latest OpenBLAS and SuiteSparse (PR
    [\#368](https://github.com/hpc4cmb/toast/pull/368)).
-   Tweaks to atmosphere simulation based on calibration campaign (PR
    [\#367](https://github.com/hpc4cmb/toast/pull/367)).
-   Add support for 2D polynomial filtering across focalplane (PR
    [\#366](https://github.com/hpc4cmb/toast/pull/366)).
-   Ground scheduler support for elevation modulated scans (PR
    [\#364](https://github.com/hpc4cmb/toast/pull/364)).
-   Add better dictionary interface to Cache class (PR
    [\#363](https://github.com/hpc4cmb/toast/pull/363)).
-   Support simulating basic non-ideal HWP response (PR
    [\#362](https://github.com/hpc4cmb/toast/pull/362)).
-   Ground scheduler support for fixed elevations and partial scans (PR
    [\#361](https://github.com/hpc4cmb/toast/pull/361)).
-   Additional check for NULL plan returned from FFTW (PR
    [\#360](https://github.com/hpc4cmb/toast/pull/360)).

## 2.3.8 (2020-06-27)

-   Minor release focusing on build system changes to support packaging
-   Update bundled pybind11 and other fixes for wheels and conda
    packages (PR [\#359](https://github.com/hpc4cmb/toast/pull/359)).

## 2.3.7 (2020-06-13)

-   Documentation updates and deployment of pip wheels on tags (PR
    [\#356](https://github.com/hpc4cmb/toast/pull/356)).
-   Cleanups of conviqt polarization support (PR
    [\#347](https://github.com/hpc4cmb/toast/pull/347)).
-   Support elevation nods in ground simulations (PR
    [\#355](https://github.com/hpc4cmb/toast/pull/355)).
-   Fix a bug in parallel writing of Healpix FITS files (PR
    [\#354](https://github.com/hpc4cmb/toast/pull/354)).
-   Remove dependence on MPI compilers. Only mpi4py is needed (PR
    [\#350](https://github.com/hpc4cmb/toast/pull/350)).
-   Use the native mapmaker by default in the example pipelines (PR
    [\#352](https://github.com/hpc4cmb/toast/pull/352)).
-   Updates to build system for pip wheel compatibility (PR
    [\#348](https://github.com/hpc4cmb/toast/pull/348),
    [\#351](https://github.com/hpc4cmb/toast/pull/351)).
-   Switch to github actions instead of travis for continuous
    integration (PR [\#349](https://github.com/hpc4cmb/toast/pull/349)).
-   Updates to many parts of the simulation and filtering operators (PR
    [\#341](https://github.com/hpc4cmb/toast/pull/341)).
-   In the default Healpix pointing matrix, support None for HWP angle
    (PR [\#345](https://github.com/hpc4cmb/toast/pull/345)).
-   Add support for HWP in conviqt beam convolution (PR
    [\#343](https://github.com/hpc4cmb/toast/pull/343)).
-   Reimplementation of example jobs used for benchmarks (PR
    [\#332](https://github.com/hpc4cmb/toast/pull/332)).
-   Apply atmosphere scaling in temperature, not intensity (PR
    [\#328](https://github.com/hpc4cmb/toast/pull/328)).
-   Minor bugfix in binner when running in debug mode (PR
    [\#325](https://github.com/hpc4cmb/toast/pull/325)).
-   Add optional boresight offset to the scheduler (PR
    [\#329](https://github.com/hpc4cmb/toast/pull/329)).
-   Implement helper tools for parsing mapmaker options (PR
    [\#321](https://github.com/hpc4cmb/toast/pull/321)).

## 2.3.6 (2020-01-19)

-   Overhaul documentation (PR
    [\#320](https://github.com/hpc4cmb/toast/pull/320)).
-   Small typo fix for conviqt operator (PR
    [\#319](https://github.com/hpc4cmb/toast/pull/319)).
-   Support high-cadence ground scan strategies and fix a bug in
    turnaround simulation (PR
    [\#316](https://github.com/hpc4cmb/toast/pull/316)).
-   Fix BLAS / LAPACK name mangling detection (PR
    [\#315](https://github.com/hpc4cmb/toast/pull/315)).
-   Allow disabling sky sim in example pipeline (PR
    [\#313](https://github.com/hpc4cmb/toast/pull/313)).

## 2.3.5 (2019-11-19)

-   Documentation updates (PR
    [\#310](https://github.com/hpc4cmb/toast/pull/310)).

## 2.3.4 (2019-11-17)

-   Disabling timing tests during build of conda package.

## 2.3.3 (2019-11-16)

-   Change way that the MPI communicator is passed to C++ (PR
    [\#309](https://github.com/hpc4cmb/toast/pull/309)).

## 2.3.2 (2019-11-13)

-   Convert atmosphere simulation to new libaatm package (PR
    [\#307](https://github.com/hpc4cmb/toast/pull/307)).
-   Improve vector math unit tests (PR
    [\#296](https://github.com/hpc4cmb/toast/pull/296)).
-   Updates to conviqt operator (PR
    [\#304](https://github.com/hpc4cmb/toast/pull/304)).
-   Satellite example pipeline cleanups.
-   Store local pixel information in the data dictionary (PR
    [\#306](https://github.com/hpc4cmb/toast/pull/306)).
-   Add elevation-dependent noise (PR
    [\#303](https://github.com/hpc4cmb/toast/pull/303)).
-   Move global / local pixel lookup into compiled code (PR
    [\#302](https://github.com/hpc4cmb/toast/pull/302)).
-   PySM operator changes to communicator (PR
    [\#301](https://github.com/hpc4cmb/toast/pull/301)).
-   Install documentation updates (PR
    [\#300](https://github.com/hpc4cmb/toast/pull/300),
    [\#299](https://github.com/hpc4cmb/toast/pull/299)).

## 2.3.1 (2019-10-14)

-   Fix bug when writing FITS maps serially.
-   Improve printing of Environment and TOD classes (PR
    [\#294](https://github.com/hpc4cmb/toast/pull/294)).
-   Fix a race condition (PR
    [\#292](https://github.com/hpc4cmb/toast/pull/292)).
-   Control the Numba backend from TOAST (PR
    [\#283](https://github.com/hpc4cmb/toast/pull/283),
    [\#291](https://github.com/hpc4cmb/toast/pull/291)).
-   Functional TOAST map-maker (PR
    [\#288](https://github.com/hpc4cmb/toast/pull/288)).
-   Large improvements to ground sim example (PR
    [\#290](https://github.com/hpc4cmb/toast/pull/290)).
-   Overhaul examples to match 2.3.0 changes (PR
    [\#286](https://github.com/hpc4cmb/toast/pull/286)).
-   Handle small angles and improve unit tests for healpix.

## 2.3.0 (2019-08-13)

-   Rewrite of internal compiled codebase and build system.
-   Move common pipeline configuration to a new module (PR
    [\#280](https://github.com/hpc4cmb/toast/pull/280)).
-   Add scan synchronous simulation operator (PR
    [\#278](https://github.com/hpc4cmb/toast/pull/278)).
