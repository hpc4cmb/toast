
SUBDIRS = 

# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast/util \
-I$(top_srcdir)/src/libtoast/math \
-I$(top_srcdir)/src/libtoast/atm \
-I$(top_srcdir)/src/libtoast/tod \
-I$(top_srcdir)/src/libtoast/fod

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS =

AM_LIBS = 

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


noinst_HEADERS = toast_fod_internal.hpp

noinst_LTLIBRARIES = libtoast_fod.la

libtoast_fod_la_includedir = $(includedir)/toast

libtoast_fod_la_include_HEADERS = \
toast/fod.hpp

libtoast_fod_la_SOURCES = \
toast_psd_tools.cpp


