
SUBDIRS = 

if HAVE_AM_MPI
else
  SUBDIRS += fakempi
endif

SUBDIRS += math
SUBDIRS += tod
SUBDIRS += map


# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast \
-I$(top_srcdir)/src/libtoast/math \
-I$(top_srcdir)/src/libtoast/tod \
-I$(top_srcdir)/src/libtoast/map

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS =

AM_LIBS = 

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


include_HEADERS = toast.hpp ctoast.h

noinst_HEADERS = toast_internal.hpp ctoast.hpp

lib_LTLIBRARIES = libtoast.la

libtoast_la_SOURCES = \
toast.cpp \
ctoast.cpp

libtoast_la_LDFLAGS = -version-info @TOAST_LT_VERSION@

libtoast_la_LIBADD = \
$(top_builddir)/src/libtoast/map/libtoast_map.la \
$(top_builddir)/src/libtoast/tod/libtoast_tod.la \
$(top_builddir)/src/libtoast/math/libtoast_math.la


if HAVE_AM_MPI
else
  libtoast_la_LIBADD += $(top_builddir)/src/libtoast/fakempi/libtoast_fakempi.la $(AM_LIBS)
endif

libtoast_la_LIBADD += $(AM_LIBS)

