
SUBDIRS = 

if HAVE_AM_MPI
else
  SUBDIRS += fakempi
endif


# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS =

AM_LIBS = 

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


EXTRA_DIST = 

include_HEADERS = toast.hpp

noinst_HEADERS = \
	toast_internal.hpp

lib_LTLIBRARIES = libtoast.la

libtoast_la_includedir = $(includedir)/toast

libtoast_la_include_HEADERS = \
    toast/memory.hpp
	

libtoast_la_SOURCES = \
toast.cpp \
toast_memory.cpp


libtoast_la_LDFLAGS = -version-info @TOAST_LT_VERSION@

libtoast_la_LIBADD =

if HAVE_AM_MPI
else
  libtoast_la_LIBADD += $(top_builddir)/src/libtoast/fakempi/libtoast_fakempi.la $(AM_LIBS)
endif

