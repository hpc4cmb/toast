#
# SYNOPSIS
#
#   ACX_MKL([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   This macro tries to link against the Intel Math Kernel Library.
#   Specifically, it tries to link to the single dynamic runtime library
#   (libmkl_rt).  The linking command depends on the compiler toolchain
#   (Intel or GNU), and the architecture.
#
#   To link an application with MKL, you should link with:
#
#   	$MKL $PTHREAD_LIBS $MATH
#
#   The user may use:
# 
#       --with-mkl=<dir>
#
#   to manually specify the directory containing the architecture specific
#   library files.
#
#   ACTION-IF-FOUND is a list of shell commands to run if the FFTW library is
#   found, and ACTION-IF-NOT-FOUND is a list of commands to run it if it is
#   not found. If ACTION-IF-FOUND is not specified, the default action will
#   define HAVE_FFTW and set output variables above.
#
#   This macro requires autoconf 2.50 or later.
#
# LAST MODIFICATION
#
#   2016-11-22
#
# COPYING
#
#   Copyright (c) 2016 Theodore Kisner <tskisner@lbl.gov>
#
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without modification,
#   are permitted provided that the following conditions are met:
#
#   o  Redistributions of source code must retain the above copyright notice, 
#      this list of conditions and the following disclaimer.
#
#   o  Redistributions in binary form must reproduce the above copyright notice, 
#      this list of conditions and the following disclaimer in the documentation
#      and/or other materials provided with the distribution.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
#   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
#   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
#   IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
#   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
#   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
#   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
#   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
#   OF THE POSSIBILITY OF SUCH DAMAGE.


AC_DEFUN([ACX_MKL], [
AC_PREREQ(2.50)
AC_REQUIRE([ACX_MATH])
AC_REQUIRE([AX_PTHREAD])

acx_mkl_ok=no
acx_mkl_ldflags=""

MKL_CPPFLAGS=""
MKL=""

AC_ARG_WITH(mkl, [AC_HELP_STRING([--with-mkl=<dir>], [use MKL architecture-specific libraries installed in <dir>.  Set to "no" to disable.])])

if test x"$with_mkl" != x; then
   if test x"$with_mkl" != xno; then
      MKL_CPPFLAGS="-I$with_mkl/../../include"
      acx_mkl_ldflags="-L$with_mkl"
   else
      acx_mkl_ok=disable
   fi
fi

if test $acx_mkl_ok = disable; then
   echo "**** MKL explicitly disabled by configure."
else

   # Save environment

   acx_mkl_save_CPPFLAGS="$CPPFLAGS"
   acx_mkl_save_LIBS="$LIBS"

   # Test serial compile and linking

   # First, try user-specified location or linked-by-default (for example using
   # a compiler wrapper script)

   MKL="$acx_mkl_ldflags -lmkl_rt -ldl"
   CPPFLAGS="$CPPFLAGS $PTHREAD_CFLAGS $MKL_CPPFLAGS"
   LIBS="$acx_mkl_save_LIBS $MKL $PTHREAD_LIBS $MATH"

   AC_CHECK_HEADERS([mkl_dfti.h])

   AC_MSG_CHECKING([for DftiCreateDescriptor in user specified location])
   AC_TRY_LINK_FUNC(DftiCreateDescriptor, [acx_mkl_ok=yes;AC_DEFINE(HAVE_MKL,1,[Define if you have the Intel MKL library.])], [])
   AC_MSG_RESULT($acx_mkl_ok)

   if test $acx_mkl_ok = no; then
      MKL_CPPFLAGS=""
      MKL=""
   fi

   # Restore environment

   LIBS="$acx_mkl_save_LIBS"
   CPPFLAGS="$acx_mkl_save_CPPFLAGS"

fi

# Define exported variables

AC_SUBST(MKL_CPPFLAGS)
AC_SUBST(MKL)

# Execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:

if test x"$acx_mkl_ok" = xyes; then
   ifelse([$1],,[echo "**** Enabling MKL support."],[$1])
else
   ifelse([$2],,[echo "**** MKL not found - disabling support."],[$2])
fi

])
