
SUBDIRS = 

# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast/util \
-I$(top_srcdir)/src/libtoast/math \
-I$(top_srcdir)/src/libtoast/atm \
-I$(top_srcdir)/src/libtoast/tod \
-I$(top_srcdir)/src/libtoast/fod \
-I$(top_srcdir)/src/libtoast/map

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS =

AM_LIBS = 

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


noinst_HEADERS = toast_map_internal.hpp

noinst_LTLIBRARIES = libtoast_map.la

libtoast_map_la_includedir = $(includedir)/toast

libtoast_map_la_include_HEADERS = \
toast/map.hpp \
toast/cov.hpp \
toast/map_tools.hpp

libtoast_map_la_SOURCES = \
toast_cov.cpp \
toast_map_tools.cpp
