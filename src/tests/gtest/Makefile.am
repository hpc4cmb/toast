
# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/tests/gtest

AM_CFLAGS =
AM_CXXFLAGS = \
-DGTEST_HAS_PTHREAD=0 \
-DGTEST_LANG_CXX11=1

AM_LDFLAGS = 

AM_LIBS = 

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


noinst_HEADERS = \
	gtest-internal-inl.h \
	gtest/gtest-death-test.h \
	gtest/gtest-message.h \
	gtest/gtest-param-test.h \
	gtest/gtest-param-test.h.pump \
	gtest/gtest-printers.h \
	gtest/gtest-spi.h \
	gtest/gtest-test-part.h \
	gtest/gtest-typed-test.h \
	gtest/gtest.h \
	gtest/gtest_pred_impl.h \
	gtest/gtest_prod.h \
	gtest/internal/gtest-death-test-internal.h \
	gtest/internal/gtest-filepath.h \
	gtest/internal/gtest-internal.h \
	gtest/internal/gtest-linked_ptr.h \
	gtest/internal/gtest-param-util-generated.h \
	gtest/internal/gtest-param-util-generated.h.pump \
	gtest/internal/gtest-param-util.h \
	gtest/internal/gtest-port-arch.h \
	gtest/internal/gtest-port.h \
	gtest/internal/gtest-string.h \
	gtest/internal/gtest-tuple.h \
	gtest/internal/gtest-tuple.h.pump \
	gtest/internal/gtest-type-util.h \
	gtest/internal/gtest-type-util.h.pump \
	gtest/internal/custom/gtest.h \
	gtest/internal/custom/gtest-port.h \
	gtest/internal/custom/gtest-printers.h


noinst_LTLIBRARIES = libtoast_gtest.la

libtoast_gtest_la_includedir = $(includedir)/gtest

libtoast_gtest_la_SOURCES = \
	gtest-all.cc

libtoast_gtest_la_LIBADD =


