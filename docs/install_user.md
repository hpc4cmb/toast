(install:user)=
# User Installation

If you are using TOAST to build simulation and analysis workflows,
including mixing built-in functionality with your own custom tools, then
you can use one of these methods to get started. If you want to hack on
the TOAST package itself, [see the section on developer installs](install:dev).

If you want to use a pre-installed version of TOAST at NERSC, [see that section](nersc:).

## Pip Binary Wheels

If you already have a newer Python3 (\>= 3.7), then you can install
pre-built TOAST packages from PyPI. You should always use virtualenv or
similar tools to manage your python environments rather than
pip-installing packages as root.

On Debian / Ubuntu Linux, you should install these minimal packages:

```{code-block} bash
apt update
apt install python3 python3-pip python3-venv
```

On Redhat / Centos you may need to enable software collections first.  See [documentation here](https://github.com/sclorg/centos-release-scl) on how to do that (it may be as easy as `yum install centos-release-scl`).  Then install a newer python3:

```{code-block} bash
yum update
yum install rh-python38
scl enable rh-python38 bash
```

On MacOS, you can use homebrew or macports to install a recent python3.
Now verify that your python is at least 3.7:

```{code-block} bash
python3 --version
```

Next create a virtualenv (name it whatever you like):

```{code-block} bash
python3 -m venv ${HOME}/cmb
```

Now activate this environment:

```{code-block} bash
source ${HOME}/cmb/bin/activate
```

Within this virtualenv, update pip to the latest version. This is needed
in order to install more recent wheels from PyPI:

```{code-block} bash
python3 -m pip install --upgrade pip
```

Next, use pip to install toast and its requirements:

```{code-block} bash
pip install toast
```

(install:user:mpi)=
### Enabling MPI Support

At this point you have toast installed and you can use it from serial
scripts and notebooks. If you want to enable effective parallelism with
toast (useful if your computer has many cores), then you need to install
the `mpi4py` package. This package requires MPI compilers (usually MPICH
or OpenMPI). Your system may already have some MPI compilers installed-
try this:

```{code-block} bash
which mpicc
mpicc -show
```

If the mpicc command is not found, you should use your OS package
manager to install the development packages for MPICH or OpenMPI. Now
you can install mpi4py:

```{code-block} bash
pip install mpi4py
```

For more details about custom installation options for mpi4py, read the
[documentation for that
package](https://mpi4py.readthedocs.io/en/stable/install.html). After installation, [you should run the unit tests](install:test)

(install:user:conda)=
## Conda Packages

If you already use (or would like to use) the conda python stack, then you can install TOAST
and all of its optional dependencies with the conda package manager. The
conda-forge ecosystem allows us to create packages that are built
consistently with all their dependencies. When we talk about the `base` conda environment (previously called the "root" environment), this is the initial environment loaded when the conda shell initialization is done.  Many people just use this base environment for everything, but this forces you to use the same versions of packages for every script and notebook across all the different projects you might be working on.  It also creates a maintenance nightmare when you need to update packages.  In this section we walk through creating a conda *environment* to use for TOAST / CMB analysis work.

By keeping a minimal base and using environments for all other work, it is trivial to update the conda tool itself and the other essential packages in base.  If one of your working environments becomes horribly out of date or broken, just delete it and make a new one.

### Using Anaconda with conda-forge Packages

If you already have Anaconda python installed, the base conda environment may already be activated by lines added to your shell resource file when you installed it.  First, check that your conda version is new enough (\>=4.9):

```{code-block} bash
conda --version
```

If this is too old, update the conda package in the base environment with:

```{code-block} bash
conda update conda
```

Next, add conda-forge to the channels which are searched for packages.  Also set the "strict" channel priority, which prevents mixing packages and dependencies between channels (which can cause many problems):

```{code-block} bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Now skip ahead to the section on [creating an environment](install:user:conda:env).

### Using a Native conda-forge Base

If you are starting from scratch, we recommend using the "miniforge" installer to set up a base environment that uses the conda-forge channel by default.  You can find the [download and install instructions here](https://github.com/conda-forge/miniforge/#download).  After following those instructions, you will have a conda base environment ready to go, with the conda-forge channel set to the default.


(install:user:conda:env)=
### Creating an Environment

Once you have the base environment activated, we are ready to create a new environment for our work.  This environment will be located in a default location in your home directory (usually inside `${HOME}/.conda/`).  You can call this whatever you like.  For this example, we use the name "cmb".  You can read more about managing conda environments [in the official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

```{code-block} bash
conda create -y -n cmb
```

Now we can activate our new (and mostly empty) environment:

```{code-block} bash
conda activate cmb
```

Finally, we can install the toast package:

```{code-block} bash
conda install toast
```

You can also install other packages you use regularly, like `jupyter`, etc.  If you want to make use of MPI with toast, you have several choices.  If you are running on a laptop or workstation, you can likely do `conda install mpi4py` and get a working installation.  If you are running on a cluster or computing center which has a specific MPI installation to use, then you can either install mpi4py with pip [(see notes above)](install:user:mpi) **or** you can [install a "fake" conda MPI package](https://conda-forge.org/docs/user/tipsandtricks.html#using-external-message-passing-interface-mpi-libraries) and then use conda to install mpi4py.  Installing mpi4py with pip using the system MPI seems like the more general solution, and is what we frequently do in production.

You can always "deactivate" this environment (and go back to the base environment) with:

```{code-block} bash
conda deactivate
```

As always, after installation, [you should run the unit tests](install:test).


## Something Else

If you have a custom install situation that is not met by the above
solutions, then you should [follow the instructions for a developer install](install:dev).

