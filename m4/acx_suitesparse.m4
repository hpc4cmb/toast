#
# SYNOPSIS
#
#   ACX_SUITESPARSE([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   This macro looks for a version of the SuiteSparse library.  The
#   SUITESPARSE_CPPFLAGS and SUITESPARSE output variables hold the compile and
#   link flags.
#
#   To link an application with SuiteSparse, you should link with:
#
#   	$SUITESPARSE
#
#   The user may use:
#
#       --with-suitesparse=<PATH>
#
#   to manually specify the SuiteSparse installation prefix.
#
#   ACTION-IF-FOUND is a list of shell commands to run if an SuiteSparse
#   library is found, and ACTION-IF-NOT-FOUND is a list of commands to run it
#   if it is not found. If ACTION-IF-FOUND is not specified, the default action
#   will define HAVE_SUITESPARSE and set output variables above.
#
#   This macro requires autoconf 2.50 or later.
#
# LAST MODIFICATION
#
#   2019-02-05
#
# COPYING
#
#   Copyright (c) 2018-2019 Theodore Kisner <tskisner@lbl.gov>
#
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   o  Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#   o  Redistributions in binary form must reproduce the above copyright notice,
#      this list of conditions and the following disclaimer in the documentation
#      and/or other materials provided with the distribution.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#   POSSIBILITY OF SUCH DAMAGE.
#

AC_DEFUN([ACX_SUITESPARSE], [
AC_PREREQ(2.50)
AC_REQUIRE([AX_CHECK_LAPACK])
AC_REQUIRE([AX_OPENMP])

acx_suitesparse_ok=no

acx_suitesparse_default="-lcholmod -lsuitesparseconfig"

SUITESPARSE_CPPFLAGS=""
SUITESPARSE_LDFLAGS=""
SUITESPARSE_LIBS="$acx_suitesparse_default"

AC_ARG_WITH(suitesparse, [AC_HELP_STRING([--with-suitesparse=<PATH>], [use the SuiteSparse installed in <PATH>.])])

if test x"$with_suitesparse" != x; then
   if test x"$with_suitesparse" != xno; then
      SUITESPARSE_CPPFLAGS="-I$with_suitesparse/include"
      SUITESPARSE_LDFLAGS="-L$with_suitesparse/lib64 -L$with_suitesparse/lib"
   else
      acx_suitesparse_ok=disable
   fi
fi

if test $acx_suitesparse_ok = disable; then
   echo "**** SuiteSparse explicitly disabled by configure."
else

   # Save environment

   acx_suitesparse_save_CPPFLAGS="$CPPFLAGS"
   acx_suitesparse_save_LIBS="$LIBS"

   CPPFLAGS="$CPPFLAGS $SUITESPARSE_CPPFLAGS"
   SUITESPARSE="$SUITESPARSE_LDFLAGS $SUITESPARSE_LIBS"
   LIBS="$SUITESPARSE $acx_suitesparse_save_LIBS $LAPACK_LIBS $BLAS_LIBS $LIBS $FCLIBS -lm $OPENMP_CXXFLAGS"

   AC_CHECK_HEADERS([cholmod.h])

   AC_MSG_CHECKING([for cholmod_start in $SUITESPARSE])
   AC_TRY_LINK_FUNC(cholmod_start, [acx_suitesparse_ok=yes;AC_DEFINE(HAVE_SUITESPARSE,1,[Define if you have the SuiteSparse library.])], [])

   AC_MSG_RESULT($acx_suitesparse_ok)

   if test $acx_suitesparse_ok = no; then
      SUITESPARSE=""
   fi

   # Restore environment

   CPPFLAGS="$acx_suitesparse_save_CPPFLAGS"
   LIBS="$acx_suitesparse_save_LIBS"

fi

# Define exported variables

AC_SUBST(SUITESPARSE_CPPFLAGS)
AC_SUBST(SUITESPARSE)

# Execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:

if test x"$acx_suitesparse_ok" = xyes; then
   ifelse([$1],,[echo "**** Enabling support for SuiteSparse."],[$1])
else
   ifelse([$2],,[echo "**** SuiteSparse not found - disabling support."],[$2])
fi

])
