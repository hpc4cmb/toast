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


toastmapdir = $(pythondir)/toast/map

toastmap_PYTHON = \
__init__.py \
pixels.py \
noise.py \
rings.py \
pysm.py \
smooth.py \
mapsampler.py

if HAVE_AM_MPI
  toastmap_PYTHON += madam.py
endif


clean-local :
	@rm -f *.pyc

