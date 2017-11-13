
SUBDIRS = 

# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast/math \
-I$(top_srcdir)/src/libtoast/atm \
-I$(top_srcdir)/src/libtoast/tod \
-I$(top_srcdir)/src/libtoast/fod \
-I$(top_srcdir)/src/libtoast/util

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS =

AM_LIBS = 

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


noinst_HEADERS = toast_util_internal.hpp

noinst_LTLIBRARIES = libtoast_util.la

libtoast_util_la_includedir = $(includedir)/toast

libtoast_util_la_include_HEADERS = \
toast/base_timer.hpp \
toast/timer.hpp \
toast/timing_manager.hpp

libtoast_util_la_SOURCES = \
toast_base_timer.cpp \
toast_timer.cpp \
toast_timing_manager.cpp



