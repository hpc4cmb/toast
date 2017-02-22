
SUBDIRS = 

# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast/math

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS =

AM_LIBS = 

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


EXTRA_DIST = Random123

noinst_HEADERS = toast_math_internal.hpp

noinst_LTLIBRARIES = libtoast_math.la

libtoast_math_la_includedir = $(includedir)/toast

libtoast_math_la_include_HEADERS = \
toast/memory.hpp \
toast/rng.hpp \
toast/qarray.hpp \
toast/math.hpp \
toast/fft.hpp \
toast/sf.hpp \
toast/healpix.hpp \
toast/lapack.hpp

libtoast_math_la_SOURCES = \
toast_math.cpp \
toast_memory.cpp \
toast_rng.cpp \
toast_qarray.cpp \
toast_fft.cpp \
toast_sf.cpp \
toast_healpix.cpp \
toast_lapack.cpp


# install / uninstall Random123 headers

install-data-local:
	@$(mkinstalldirs) $(DESTDIR)$(includedir)/Random123; \
	cp -r $(srcdir)/Random123/* $(DESTDIR)$(includedir)/Random123/; \
	chmod -R a=rX,u+w $(DESTDIR)$(includedir)/Random123

uninstall-local:
	@rm -rf $(DESTDIR)$(includedir)/Random123; \
	rm -rf $(DESTDIR)$(includedir)/toast


