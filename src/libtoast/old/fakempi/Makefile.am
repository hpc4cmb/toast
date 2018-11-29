
# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast/fakempi

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS =

AM_LIBS = 

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


EXTRA_DIST = COPYRIGHT_MPICH README.md

include_HEADERS = mpi.h

noinst_LTLIBRARIES = libtoast_fakempi.la

libtoast_fakempi_la_SOURCES = \
mpi.c

