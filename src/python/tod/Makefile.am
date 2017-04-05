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


toasttoddir = $(pythondir)/toast/tod

toasttod_PYTHON = \
__init__.py \
tod.py \
tod_math.py \
pointing_math.py \
pointing.py \
noise.py \
interval.py \
sim_interval.py \
sim_noise.py \
sim_tod.py \
sim_det_noise.py \
sim_det_map.py \
sim_det_dipole.py \
sim_det_atm.py \
sim_focalplane.py


if HAVE_AM_MPI
  toasttod_PYTHON += conviqt.py
endif


EXTRA_DIST = conviqt_example.txt

clean-local :
	@rm -f *.pyc

