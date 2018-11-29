
SUBDIRS =

# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast/util \
-I$(top_srcdir)/src/libtoast/math \
-I$(top_srcdir)/src/libtoast/atm \
-I$(top_srcdir)/src/libtoast/tod

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS =

AM_LIBS =

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


noinst_HEADERS = toast_tod_internal.hpp

noinst_LTLIBRARIES = libtoast_tod.la

libtoast_tod_la_includedir = $(includedir)/toast

libtoast_tod_la_include_HEADERS = \
toast/tod.hpp \
toast/sim_noise.hpp \
toast/pointing.hpp \
toast/filter.hpp \
toast/sim_map.hpp

libtoast_tod_la_SOURCES = \
toast_sim_noise.cpp \
toast_pointing.cpp \
toast_filter.cpp \
toast_sim_map.cpp
