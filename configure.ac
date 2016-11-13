dnl
dnl This is the Autoconf file for TOAST
dnl
dnl +------------------------
dnl | Initialize package info
dnl +------------------------
AC_INIT([TOAST], [2.2.0], [work@theodorekisner.com], [toast], [https://github.com/hpc4cmb/toast])
AC_CONFIG_SRCDIR([Makefile.am])
AM_INIT_AUTOMAKE([foreign])
AC_CONFIG_HEADERS(config.h)
AC_CONFIG_MACRO_DIR([m4])
dnl +-------------------------
dnl | If the library source code has changed at all since the last update, 
dnl | then increment revision (c:r:a becomes c:r+1:a).  If any interfaces 
dnl | have been added, removed, or changed since the last update, increment
dnl | current, and set revision to 0.  If any interfaces have been added 
dnl | since the last public release, then increment age.  If any interfaces 
dnl | have been removed since the last public release, then set age to 0.
dnl +-------------------------
TOAST_LT_VERSION="1:0:0"
AC_SUBST(TOAST_LT_VERSION)
dnl +---------------------
dnl | Compiler features
dnl +---------------------
dnl Set default optimization to '-O3' instead of the default '-g -O2'
if test x"${CFLAGS}" = x; then
   CFLAGS="-O3"
fi
if test x"${CXXFLAGS}" = x; then
   CXXFLAGS="-O3"
fi
if test x"${FCFLAGS}" = x; then
   FCFLAGS="-O3"
fi
AC_CANONICAL_HOST
AC_PROG_INSTALL
LT_INIT
dnl +-------------------------
dnl | Shared library extension
dnl +-------------------------
TOAST_SHARED_EXT=""
case $host_os in
  darwin* )
    TOAST_SHARED_EXT="dylib"
    ;;
  linux*)
    TOAST_SHARED_EXT="so"
    ;;
  *BSD*)
    TOAST_SHARED_EXT="so"
    ;;
  *)
    TOAST_SHARED_EXT="so"
    AC_MSG_WARN([Your platform was not detected- using Linux defaults for shared library suffix])
    ;;
esac
AC_SUBST(TOAST_SHARED_EXT)
dnl +-------------------------
dnl | C support
dnl +-------------------------
AC_PROG_CC
dnl +-------------------------
dnl | C++ support
dnl +-------------------------
AC_PROG_CXX
AX_CXX_COMPILE_STDCXX([11], [noext], [mandatory])
dnl +------------------------------------------------
dnl | This is a C++ project...
dnl +------------------------------------------------
AC_LANG([C++])
dnl +------------------------------------------------
dnl | Check for OpenMP
dnl +------------------------------------------------
toast_openmp=yes
ax_have_openmp=no
AC_ARG_ENABLE(openmp, [AC_HELP_STRING([--disable-openmp], [Disable OpenMP])], [toast_openmp=$enable_openmp])
if test $toast_openmp != no; then
  AX_OPENMP([ax_have_openmp=yes;AC_DEFINE(HAVE_OPENMP,1,[Define if OpenMP is enabled])])
fi
AC_SUBST([OPENMP_CXXFLAGS])
AM_CONDITIONAL([HAVE_AM_OPENMP], [test $ax_have_openmp = yes])
dnl +------------------------------------------------
dnl | Check for MPI
dnl +------------------------------------------------
toast_mpi=yes
AC_ARG_ENABLE([mpi], [AC_HELP_STRING([--disable-mpi], [Disable all checks for MPI])], [toast_mpi=$enable_mpi], [toast_mpi=yes])
if test $toast_mpi = yes; then
  ACX_MPI
  if test x"$MPICXX" = x; then
    toast_mpi=no
  else
    AC_LANG_PUSH([C])
    ACX_MPI
    AC_LANG_POP([C])
    if test x"$MPICC" = x; then
      toast_mpi=no
    fi
  fi
fi
AM_CONDITIONAL([HAVE_AM_MPI], [test toast_mpi = yes])
dnl +-------------------------
dnl | Should we enable expensive debugging checks?
dnl +-------------------------
toast_debug=no
AC_ARG_ENABLE([debug], [AC_HELP_STRING([--enable-debug], [Enable expensive debugging checks])])
if test x"$enable_debug" = x"yes"; then
  toast_debug=yes
fi
AM_CONDITIONAL([ENABLE_AM_DEBUG], [test $toast_debug = yes])
dnl +------------------------------------------------
dnl | Check for Python
dnl +------------------------------------------------
AC_LANG_PUSH([C])
toast_python=yes
AC_ARG_ENABLE(python, [AC_HELP_STRING([--disable-python], [Disable python bindings])], [toast_python=$enable_python])
if test $toast_python = yes; then
  AM_PATH_PYTHON
fi
AC_LANG_POP([C])
AM_CONDITIONAL([HAVE_AM_PYTHON], [test $toast_python = yes])
dnl +------------------------------------------------
dnl | Set outputs
dnl +------------------------------------------------
AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([src/Makefile])
#AC_CONFIG_FILES([src/libtoast/Makefile])
#AC_CONFIG_FILES([src/libtoast/fakempi/Makefile])
#AC_CONFIG_FILES([src/libtoast/math/Makefile])
#AC_CONFIG_FILES([src/libtoast/tod/Makefile])
#AC_CONFIG_FILES([src/python/Makefile])
AC_CONFIG_FILES([src/tests/gtest/Makefile])
AC_CONFIG_FILES([src/tests/Makefile])
dnl +------------------------------------------------
dnl | Generate Makefiles
dnl +------------------------------------------------
AC_OUTPUT
dnl +------------------------------------------------
dnl | Print out detected build options
dnl +------------------------------------------------
AC_MSG_NOTICE([                                           ])
AC_MSG_NOTICE([=========== Build Configuration ===========])
AC_MSG_NOTICE([                                           ])
AC_MSG_NOTICE([  Serial C++ Compiler       : $CXX])
AC_MSG_NOTICE([  C++ Compile Flags         : $CXXFLAGS])
AC_MSG_NOTICE([  Serial C Compiler         : $CC])
AC_MSG_NOTICE([  C Compile Flags           : $CFLAGS])
if test x"${ax_have_openmp}" = xyes; then
  AC_MSG_NOTICE([  OpenMP                    : Detected ($OPENMP_CXXFLAGS)])
else
  AC_MSG_NOTICE([  OpenMP                    : Disabled])
fi
if test $toast_mpi = yes; then
  AC_MSG_NOTICE([  MPI C++ Compiler          : $MPICXX])
  AC_MSG_NOTICE([  MPI C Compiler            : $MPICC])
else
  AC_MSG_NOTICE([  MPI C++ Compiler          : Not found, using fake MPI])
  AC_MSG_NOTICE([  MPI C Compiler            : Not found, using fake MPI])
fi
if test $toast_python = yes; then
  AC_MSG_NOTICE([  Python                    : $PYTHON])
else
  AC_MSG_NOTICE([  Python                    : Disabled])
fi
AC_MSG_NOTICE([                                           ])
