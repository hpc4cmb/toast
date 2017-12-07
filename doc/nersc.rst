.. _nersc:

Using at NERSC
====================

To use TOAST at NERSC, you need to have a Python3 software stack with all dependencies installed.  There is already such a software stack installed on edison and cori.


Module Files
---------------

To get access to the needed module files, add the machine-specific module file location to your search path::

    %> module use /global/common/software/cmb/${NERSC_HOST}/modulefiles

You can safely put the above line in your ~/.bashrc.ext inside the sections for edison and cori.


Load Dependencies
--------------------

In order to load a full python-3.5 stack, and also all dependencies needed by toast, do::

    %> module load toast-deps


Install TOAST
------------------

The TOAST codebase is evolving daily, therefore we do not maintain a `toast` module.
You have to install TOAST yourself from source.

When installing *any* software at NERSC, we need to
keep several things in mind:

    *  The home directories are small.

    *  The performance of the home directories when accessed by many processes
       is very bad.

    *  The scratch space has lots of room, and very good performance.

    *  Any file on the scratch space that has not be accessed for some number of
       weeks will be deleted.

So unfortunately there is no location which has good performance and also
persistent file storage.  For this example, we will install software to scratch
and assume that we will be using the software frequently enough that it will never
be purged.  If you have not used the tools for a month or so, you should probably
reinstall just to be sure that everything is in place.

First, we pick a location to install our software.  For this example, we will
be installing to a "software" directory in our scratch space.  First make sure
that exists::

    %> mkdir -p ${SCRATCH}/software

Now we will create a small shell function that loads this location into our search
paths for executables and python packages.  Add this function to ~/.bashrc.ext and
you can rename it to whatever you like::

    loadtoast () {
        export PREFIX=${SCRATCH}/software/toast
        export PATH=$PREFIX/bin:${PATH}
        export PYTHONPATH=$PREFIX/lib/python3.5/site-packages:${PYTHONPATH}
    }

Log out and back in to make this function visible to your shell environment.
Now checkout the toast source in your home directory somewhere::

    %> cd
    %> git clone https://github.com/hpc4cmb/toast

Then configure and build the software.  Unless you know what you are doing, you
should probably use the platform config example for the machine you are building
for, consider that the `toast-deps` environment requires Intel compilers::

    %> cd toast
    %> ./autogen.sh
    %> ./platforms/edison-intel.sh --prefix=${SCRATCH}/software/toast

Now we can run our function to load this installation into our environment::

    %> loadtoast

On NERSC systems, MPI is not allowed to be run on the login nodes.  In order to
run our unittests, we first get an interactive compute node::

    %> salloc

and then run the tests::

    %> srun python -c "import toast; toast.test()"

You should read through the many good NERSC webpages that describe how to use the
different machines.  There are `pages for edison <http://www.nersc.gov/users/computational-systems/edison/running-jobs/>`_
and `pages for cori <http://www.nersc.gov/users/computational-systems/cori/running-jobs/>`_.


Install Experiment Packages
------------------------------------------

If you are a member of Planck, Core, or LiteBIRD, you can get access to separate
git repos with experiment-specific scripts and tools.  You can install these to
the same location as toast.  All of those packages currently use distutils, and
you will need to do the installation from a compute node (since importing the
toast python module will load MPI)::

    %> cd toast-<experiment>
    %> salloc
    %> srun python setup.py clean
    %> srun python setup.py install --prefix=${SCRATCH}/software/toast

