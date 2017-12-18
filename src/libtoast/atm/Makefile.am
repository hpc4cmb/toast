
SUBDIRS =

# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast/util \
-I$(top_srcdir)/src/libtoast/math \
-I$(top_srcdir)/src/libtoast/atm

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS =

AM_LIBS =

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


noinst_HEADERS = toast_atm_internal.hpp

noinst_LTLIBRARIES = libtoast_atm.la

libtoast_atm_la_includedir = $(includedir)/toast

libtoast_atm_la_include_HEADERS = \
toast/atm.hpp

libtoast_atm_la_SOURCES = \
toast_atm.cpp toast_atm_util.cpp
