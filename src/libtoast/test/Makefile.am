
SUBDIRS = gtest

# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast \
-I$(top_srcdir)/src/libtoast/util \
-I$(top_srcdir)/src/libtoast/math \
-I$(top_srcdir)/src/libtoast/atm \
-I$(top_srcdir)/src/libtoast/tod \
-I$(top_srcdir)/src/libtoast/fod \
-I$(top_srcdir)/src/libtoast/map \
-I$(top_srcdir)/src/libtoast/test/gtest \
-I$(top_srcdir)/src/libtoast/test

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS =

AM_LIBS =


# Add check flags

AM_CPPFLAGS += $(CHECK_CFLAGS)
AM_LIBS += $(CHECK_LIBS)

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


EXTRA_DIST = gen_healpix_data.py data_healpix.cpp

noinst_HEADERS = toast_test.hpp

noinst_LTLIBRARIES = libtoast_test.la

libtoast_test_la_SOURCES = \
toast_test_timing.cpp \
toast_test_qarray.cpp \
toast_test_rng.cpp \
toast_test_sf.cpp \
toast_test_healpix.cpp \
toast_test_fft.cpp \
toast_test_cov.cpp \
toast_test_mpi_shmem.cpp \
toast_test_polyfilter.cpp \
toast_test_runner.cpp

libtoast_test_la_LIBADD = \
$(top_builddir)/src/libtoast/test/gtest/libtoast_gtest.la

