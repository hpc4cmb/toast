(nersc:)=
# Using TOAST at NERSC

A recent version of TOAST is already installed at NERSC, along with all
necessary dependencies. You can use this installation directly, or use
it as the basis for your own development.

## Module Files

To get access to the needed module files, add the machine-specific
module file location to your search path:

    module use /global/common/software/cmb/${NERSC_HOST}/default/modulefiles

The [default]{.title-ref} part of this path is a symlink to the latest
stable installation. There are usually several older versions kept here
as well.

You can safely put the above line in your \~/.bashrc.ext inside the
section for cori. It does not actually load anything into your
environment.

## Loading the Software

To load the software do the following:

    module load cmbenv
    source cmbenv

Note that the \"source\" command above is not \"reversible\" like normal
module operations. This is required in order to activate the underlying
conda environment. After running the above commands, TOAST and many
other common software tools will be in your environment, including a
Python3 stack.

## Installing TOAST (Optional)

The cmbenv stack contains a recent version of TOAST, but if you want to
build your own copy then you can use the cmbenv stack as a starting
point. Here are the steps:

1.  Decide on the installation location. You should install software
    either to one of the project software spaces in
    [/global/common/software]{.title-ref} or in your home directory. If
    you plan on using this installation for large parallel jobs, you
    should install to [/global/common/software]{.title-ref}.

2.  Load the cmbenv stack.

3.  Go into your git checkout of TOAST and make a build directory:

        cd toast
        mkdir build
        cd build

4.  Use the cori-intel platform file to build TOAST and install:

        ../platforms/cori-intel.sh \
        -DCMAKE_INSTALL_PREFIX=/path/to/somewhere
        make -j 4 install

5.  Set up a shell function in [\~/.bashrc.ext]{.title-ref} to load this
    into your environment search paths before the cmbenv stack:

        load_toast () {
            dir=/path/to/your/install
            export PATH="${dir}/bin:${PATH}"
            pysite=$(python3 --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
            export PYTHONPATH="${dir}/lib/python${pysite}/site-packages:${PYTHONPATH}"
        }

Now whenever you want to override the cmbenv TOAST installation you can
just do:

    load_toast
