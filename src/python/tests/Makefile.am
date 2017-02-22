SUBDIRS = 

# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast \
$(PYTHON_CPPFLAGS)

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS = $(PYTHON_LDFLAGS)

AM_LIBS = $(PYTHON_LIBS)

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


toasttestsdir = $(pythondir)/toast/tests

toasttests_PYTHON = \
__init__.py \
runner.py \
ctoast.py \
cbuffer.py \
mpi.py \
cache.py \
dist.py \
rng.py \
qarray.py \
intervals.py \
ops_madam.py


clean-local :
	@rm -f *.pyc

