
SUBDIRS = gtest

# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast \
-I$(top_srcdir)/src/tests/gtest \
-I$(top_srcdir)/src/tests

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS = 

AM_LIBS = \
$(top_builddir)/src/tests/gtest/libtoast_gtest.la

# \
#$(top_builddir)/src/libtoast/libtoast.la

# Add check flags

AM_CPPFLAGS += $(CHECK_CFLAGS)
AM_LIBS += $(CHECK_LIBS)

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


noinst_HEADERS = toast_test.hpp

bin_PROGRAMS = toast_test
TESTS = toast_test

toast_test_SOURCES = \
toast_test.cpp

toast_test_LDADD = $(AM_LIBS)


clean-local :
	@rm -rf *.out

