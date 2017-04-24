# External Software Stack

## Introduction

This package contains scripts for installing conda and all compiled
dependencies needed by the TOAST from scratch.  The rules to install
or build each dependency is separated into a file, and used by both
regular installs and building docker images.


## Configuration

Create or edit a file in the "conf" subdirectory that is named after the 
system you are building on.  This file will define compilers, flags, etc.
Optionally create files with the same name and the ".module" and ".sh"
suffixes.  These optional files should contain any modulefile and shell 
commands needed to set up the environment.  See existing files for 
examples.

To create a config for a docker image, the config file must be prefixed
with "docker-".  You should not have any "*.module" or "*.sh" files for
a docker config.


## Generate the Script

Set the CONFIG, PREFIX, and (optionally) the VERSION and MODULEDIR environment 
variables.  Then create the script with::

    $> make script

To clean up all generated scripts, do::

    $> make clean

For normal installs, this creates an install script and corresponding
module files.  For docker builds, a Dockerfile is created.  As an example,
suppose we are installing the software stack into our scratch directory
on edison.nersc.gov using the gcc config::

    $> PREFIX=${SCRATCH}/software/toast-gcc CONFIG=edison-gcc make clean
    $> PREFIX=${SCRATCH}/software/toast-gcc CONFIG=edison-gcc make script

If you don't have the $VERSION environment variable set, then a version
string based on the git revision history is used.  If you don't have the
$MODULEDIR environment variable set, then the modulefiles will be installed
to $PREFIX/modulefiles.


## Installation

For normal installs, simply run the install script.  This installs the
software and modulefile, as well as a module version file named
".version_$VERSION" in the module install directory.  You can manually
move this into place if and when you want to make that the default
version.  You can run the install script from an alternate build 
directory.  

For docker installs, run docker build from the same directory as the 
generated Dockerfile, so that the path to data files can be found.  Making 
docker images requires a working docker installation and also an Intel based 
processor if you are building an image that uses Intel python packages.  You
should familiarize yourself with the docker tool before attempting to use
it here.

As an example, suppose we want to install the script we made in the
previous section for edison.  We'll make a temporary directory on
scratch to do the building, since it is going to download and compile
several big packages.  We'll also dump all output to a log file so that
we can look at it afterwards if there are any problems::

    $> cd $SCRATCH
    $> mkdir build
    $> cd build
    $> /path/to/git/toast/external/install_edison-gcc.sh >log 2>&1 &
    $> tail -f log

After installation, the $PREFIX directory will contain directories
and files::

    $PREFIX/toast-deps/$VERSION_conda
    $PREFIX/toast-deps/$VERSION_aux
    $PREFIX/modulefiles/toast-deps/$VERSION
    $PREFIX/modulefiles/toast-deps/.version_$VERSION

If you want to make this version of toast-deps the default, then just
do::

    $> ln -s .version_$VERSION .version

