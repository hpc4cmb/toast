SUBDIRS = 

# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast \
-I$(top_srcdir)/src/libtoast/math \
-I$(top_srcdir)/src/libtoast/tod \
-I$(top_srcdir)/src/libtoast/map \
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
interval.py


clean-local :
	@rm -f *.pyc

