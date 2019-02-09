
SUBDIRS =

if HAVE_AM_MPI
else
  SUBDIRS += fakempi
endif

SUBDIRS += util
SUBDIRS += math

if HAVE_AM_SUITESPARSE
  SUBDIRS += atm
endif

SUBDIRS += tod
SUBDIRS += fod
SUBDIRS += map
SUBDIRS += test
SUBDIRS += capi


# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast \
-I$(top_srcdir)/src/libtoast/util \
-I$(top_srcdir)/src/libtoast/math \
-I$(top_srcdir)/src/libtoast/atm \
-I$(top_srcdir)/src/libtoast/tod \
-I$(top_srcdir)/src/libtoast/fod \
-I$(top_srcdir)/src/libtoast/map \
-I$(top_srcdir)/src/libtoast/test \
-I$(top_srcdir)/src/libtoast/test/gtest \
-I$(top_srcdir)/src/libtoast/capi

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS =

AM_LIBS =

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


include_HEADERS = toast.hpp

noinst_HEADERS = toast_internal.hpp

lib_LTLIBRARIES = libtoast.la

libtoast_la_SOURCES = \
toast.cpp

libtoast_la_LDFLAGS = -version-info @TOAST_LT_VERSION@

libtoast_la_LIBADD = \
$(top_builddir)/src/libtoast/capi/libtoast_capi.la \
$(top_builddir)/src/libtoast/test/libtoast_test.la \
$(top_builddir)/src/libtoast/map/libtoast_map.la \
$(top_builddir)/src/libtoast/fod/libtoast_fod.la \
$(top_builddir)/src/libtoast/tod/libtoast_tod.la \
$(top_builddir)/src/libtoast/util/libtoast_util.la

if HAVE_AM_SUITESPARSE
  libtoast_la_LIBADD += $(top_builddir)/src/libtoast/atm/libtoast_atm.la
endif

libtoast_la_LIBADD += $(top_builddir)/src/libtoast/math/libtoast_math.la

if HAVE_AM_MPI
else
  libtoast_la_LIBADD += $(top_builddir)/src/libtoast/fakempi/libtoast_fakempi.la
endif

libtoast_la_LIBADD += $(AM_LIBS)


bin_PROGRAMS = toast_test
TESTS = toast_test

toast_test_SOURCES = \
toast_test.cpp

toast_test_LDADD = \
libtoast.la \
$(AM_LIBS)


bin_PROGRAMS : libtoast.la
