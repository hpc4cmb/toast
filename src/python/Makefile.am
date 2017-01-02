SUBDIRS =

# Initialize compile and linking flags

AM_CPPFLAGS = \
-I$(top_srcdir)/src/libtoast \
-I$(top_srcdir)/src/libtoast/math \
-I$(top_srcdir)/src/libtoast/tod \
-I$(top_srcdir)/src/libtoast/map

AM_CFLAGS =
AM_CXXFLAGS =

AM_LDFLAGS =

AM_LIBS = 

# Append compile and linking flags based on configure variables
include $(top_srcdir)/src/build_options.am


pkgtoastdir = $(pyexecdir)/toast

pkgtoast_PYTHON = \
__init__.py \
ctoast.py

ctoast.py : ctoast.py.in
	@./gen_python.sh \
	"$(DESTDIR)$(libdir)/libtoast.$(TOAST_SHARED_EXT)"


EXTRA_DIST = gen_python.sh ctoast.py.in

CLEANFILES = ctoast.py

clean-local :
	@rm -f *.pyc

