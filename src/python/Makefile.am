SUBDIRS = 

#if HAVE_AM_MPI
#else
#  SUBDIRS += mpi4py
#endif

SUBDIRS += tod fod map tests

HAVE_MPI4PY = 
if HAVE_AM_MPI
  HAVE_MPI4PY += 1
endif

# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast/capi \
-I$(top_srcdir)/src/libtoast \
$(PYTHON_CPPFLAGS)

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS = $(PYTHON_LDFLAGS)

AM_LIBS = $(PYTHON_LIBS)

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


toastdir = $(pythondir)/toast

toast_PYTHON = \
__init__.py \
_version.py \
ctoast.py \
mpi.py \
fakempi.py \
cache.py \
dist.py \
rng.py \
fft.py \
qarray.py \
op.py \
healpix.py \
vis.py \
timing.py


toast_LTLIBRARIES = cbuffer.la
cbuffer_la_SOURCES = cbuffer.c

# make sure that the C++ compiler is used for linking
nodist_EXTRA_cbuffer_la_SOURCES = fake.cpp

cbuffer_la_LDFLAGS = -no-undefined -avoid-version -module -shared -export-dynamic
cbuffer_la_LIBADD = $(top_builddir)/src/libtoast/libtoast.la $(AM_LIBS)


ctoast.py : ctoast.py.in
	@./gen_python.sh $< $@ \
	"$(VERSION)" \
	"$(DESTDIR)$(libdir)/libtoast.$(TOAST_SHARED_EXT)" \
	"$(HAVE_MPI4PY)"

_version.py : _version.py.in
	@./gen_python.sh $< $@ \
	"$(VERSION)" \
	"$(DESTDIR)$(libdir)/libtoast.$(TOAST_SHARED_EXT)" \
	"$(HAVE_MPI4PY)"

mpi.py : mpi.py.in
	@./gen_python.sh $< $@ \
	"$(VERSION)" \
	"$(DESTDIR)$(libdir)/libtoast.$(TOAST_SHARED_EXT)" \
	"$(HAVE_MPI4PY)"


EXTRA_DIST = gen_python.sh ctoast.py.in _version.py.in mpi.py.in

CLEANFILES = ctoast.py _version.py mpi.py

clean-local :
	@rm -f *.pyc *.pyo

