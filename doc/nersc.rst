.. _nersc:

Using at NERSC
====================

To use TOAST at NERSC, you need to have a Python3 software stack with all dependencies installed.  There is already such a software stack installed on edison and cori.


Module Files
---------------

To get access to the needed module files, add the machine-specific module file location to your search path::

    %> module use /global/common/${NERSC_HOST}/contrib/hpcosmo/modulefiles

You can safely put the above line in your ~/.bashrc.ext inside the sections for edison and cori.


Load Dependencies
--------------------

In order to load a full python-3.5 stack, and also all dependencies needed by toast, do::

    %> module load toast-deps


Install TOAST
------------------

When installing *any* software at NERSC, we need to keep several things in mind:

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
be installing to a "software" directory in our scratch space.  First create
these directories::

    %> mkdir -p ${SCRATCH}/software/toast/bin
    %> mkdir -p ${SCRATCH}/software/toast/lib/python3.5/site-packages

Now we will create a small shell function that loads this location into our search
paths for executables and python packages.  Add this function to ~/.bashrc.ext and
you can rename it to whatever you like::

    toast () {
        pref=${SCRATCH}/software/toast
        export PATH=${pref}/bin:${PATH}
        export PYTHONPATH=${pref}/lib/python3.5/site-packages:${PYTHONPATH}
    }

Log out and back in to make this function visible to your shell environment.
Now checkout the toast source in your home directory somewhere::

    %> cd
    %> git clone https://github.com/hpc4cmb/toast.git

Now we can run our function::

    %> toast

And now we can install toast.  On NERSC systems, MPI is not allowed to be run on
the login nodes.  Since toast uses MPI, we have to set a special environment 
variable when running setup.py.  This will disable MPI inside toast during installation::

    %> TOAST_NO_MPI=1 python3 setup.py install --prefix=${SCRATCH}/software/toast

Whenever you update your git checkout of toast (or if you have not used toast for
a month or so), just re-run the above command to update your installation.


Install Experiment Packages
------------------------------------------

If you are a member of Planck, Core, or LiteBIRD, you can get access to separate
git repos with experiment-specific scripts and tools.  You can install these to
the same location as toast using the same command above.  Re-run the install
command when you update your git checkout.

