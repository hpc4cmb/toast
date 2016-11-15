#
# SYNOPSIS
#
#   ACX_FFTW([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   This macro looks for a version of FFTW3.  The FFTW_CPPFLAGS and FFTW
#   output variables hold the compile and link flags.
#
#   To link an application with FFTW, you should link with:
#
#   	$FFTW
#
#   The user may use:
# 
#       --with-fftw=<dir>
#
#   to manually specify the installed prefix of FFTW.
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
#   2016-11-15
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


AC_DEFUN([ACX_FFTW], [
AC_PREREQ(2.50)
AC_REQUIRE([ACX_MATH])
AC_REQUIRE([AX_PTHREAD])

acx_fftw_ok=no
acx_fftw_threads=no
acx_fftw_ldflags=""
acx_fftw_libs="-lfftw3"
acx_fftw_libs_threads="-lfftw3_threads"

FFTW_CPPFLAGS=""
FFTW=""

AC_ARG_WITH(fftw, [AC_HELP_STRING([--with-fftw=<dir>], [use FFTW installed in prefix <dir>.  Set to "no" to disable.])])

if test x"$with_fftw" != x; then
   if test x"$with_fftw" != xno; then
      FFTW_CPPFLAGS="-I$with_fftw/include"
      acx_fftw_ldflags="-L$with_fftw/lib"
   else
      acx_fftw_ok=disable
   fi
fi

if test $acx_fftw_ok = disable; then
   echo "**** FFTW explicitly disabled by configure."
else

   # Save environment

   acx_fftw_save_CPPFLAGS="$CPPFLAGS"
   acx_fftw_save_LIBS="$LIBS"

   # Test serial compile and linking

   # First, try user-specified location or linked-by-default (for example using
   # a compiler wrapper script)

   FFTW="$acx_fftw_ldflags $acx_fftw_libs"
   CPPFLAGS="$CPPFLAGS $FFTW_CPPFLAGS"
   LIBS="$FFTW $acx_fftw_save_LIBS $MATH"

   AC_CHECK_HEADERS([fftw3.h])

   AC_MSG_CHECKING([for fftw_malloc in user specified location])
   AC_TRY_LINK_FUNC(fftw_malloc, [acx_fftw_ok=yes;AC_DEFINE(HAVE_FFTW,1,[Define if you have the FFTW library.])], [])
   AC_MSG_RESULT($acx_fftw_ok)

   if test $acx_fftw_ok = yes; then
      FFTW="$acx_fftw_ldflags $acx_fftw_libs_threads $acx_fftw_libs"
      CPPFLAGS="$CPPFLAGS $PTHREAD_CFLAGS"
      LIBS="$FFTW $acx_fftw_save_LIBS $MATH $PTHREAD_LIBS"

      AC_MSG_CHECKING([for fftw_plan_with_nthreads in user specified location])
      AC_TRY_LINK_FUNC(fftw_plan_with_nthreads, [acx_fftw_threads_ok=yes;AC_DEFINE(HAVE_FFTW_THREADS,1,[Define if you have the FFTW threads library.])], [])
      AC_MSG_RESULT($acx_fftw_threads_ok)
   fi

   if test $acx_fftw_ok = no; then
      # try pkg-config
      if test `pkg-config --exists fftw`; then
         FFTW_CPPFLAGS=`pkg-config --cflags fftw`
         FFTW=`pkg-config --libs fftw`
         CPPFLAGS="$acx_fftw_save_CPPFLAGS $FFTW_CPPFLAGS"
         LIBS="$FFTW $acx_fftw_save_LIBS $MATH"
	
         AC_CHECK_HEADERS([fftw3.h])

         AC_MSG_CHECKING([for fftw_malloc in pkg-config location])
         AC_TRY_LINK_FUNC(fftw_malloc, [acx_fftw_ok=yes;AC_DEFINE(HAVE_FFTW,1,[Define if you have the FFTW library.])], [])
         AC_MSG_RESULT($acx_fftw_ok)

         if test $acx_fftw_ok = yes; then
            CPPFLAGS="$CPPFLAGS $PTHREAD_CFLAGS"
            LIBS="$acx_fftw_libs_threads $FFTW $acx_fftw_save_LIBS $MATH $PTHREAD_LIBS"
            AC_MSG_CHECKING([for fftw_plan_with_nthreads in pkg-config location])
            AC_TRY_LINK_FUNC(fftw_plan_with_nthreads, [acx_fftw_threads_ok=yes;AC_DEFINE(HAVE_FFTW_THREADS,1,[Define if you have the FFTW threads library.])], [])
            AC_MSG_RESULT($acx_fftw_threads_ok)

            if test $acx_fftw_threads_ok = yes; then
               FFTW="$acx_fftw_libs_threads $FFTW"
            fi
         fi

      fi

   fi

   if test $acx_fftw_ok = no; then
      FFTW_CPPFLAGS=""
      FFTW=""
   fi

   # Restore environment

   LIBS="$acx_fftw_save_LIBS"
   CPPFLAGS="$acx_fftw_save_CPPFLAGS"

fi

# Define exported variables

AC_SUBST(FFTW_CPPFLAGS)
AC_SUBST(FFTW)

# Execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:

if test x"$acx_fftw_ok" = xyes; then
   ifelse([$1],,[echo "**** Enabling FFTW support."],[$1])
else
   ifelse([$2],,[echo "**** FFTW not found - disabling support."],[$2])
fi

])
