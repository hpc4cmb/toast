
SUBDIRS = 

# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast/util \
-I$(top_srcdir)/src/libtoast/math \
-I$(top_srcdir)/src/libtoast/atm \
-I$(top_srcdir)/src/libtoast/tod \
-I$(top_srcdir)/src/libtoast/fod \
-I$(top_srcdir)/src/libtoast/map \
-I$(top_srcdir)/src/libtoast/test \
-I$(top_srcdir)/src/libtoast/test/gtest \
-I$(top_srcdir)/src/libtoast/capi \
-I$(top_srcdir)/src/libtoast

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS =

AM_LIBS = 

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am

include_HEADERS = ctoast.h

noinst_HEADERS = ctoast.hpp

noinst_LTLIBRARIES = libtoast_capi.la

libtoast_capi_la_SOURCES = \
ctoast.cpp


