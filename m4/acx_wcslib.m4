#
# SYNOPSIS
#
#   ACX_WCSLIB([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   This macro looks for the wcslib (http://www.atnf.csiro.au/people/mcalabre/WCS/)
#   library.  By default, it checks user options and then tries standard names.
#   The WCSLIB_CPPFLAGS and WCSLIB output variables hold the compile and link flags.
#
#   To link an application with wcslib, you should link with:
#
#   	$WCSLIB $MATH
#
#   The user may use:
# 
#       --with-wcslib=<dir>
#
#   to manually specify the installed prefix of wcslib.
#
#   ACTION-IF-FOUND is a list of shell commands to run if the wcslib library is
#   found, and ACTION-IF-NOT-FOUND is a list of commands to run it if it is
#   not found. If ACTION-IF-FOUND is not specified, the default action will
#   define HAVE_WCSLIB and set output variables above.
#
#   This macro requires autoconf 2.50 or later.
#
# LAST MODIFICATION
#
#   2017-01-01
#
# COPYING
#
#   Copyright (c) 2013-2017 Theodore Kisner <tskisner@lbl.gov>
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
#

AC_DEFUN([ACX_WCSLIB], [
AC_PREREQ(2.50)
AC_REQUIRE([ACX_MATH])

acx_wcs_ok=no
acx_wcs_ldflags=""
acx_wcs_libs="-lwcs"

WCSLIB_CPPFLAGS=""
WCSLIB=""

AC_ARG_WITH(wcslib, [AC_HELP_STRING([--with-wcslib=<dir>], [use wcslib installed in prefix <dir>.  Set to "no" to disable.])])

if test x"$with_wcslib" != x; then
   if test x"$with_wcslib" != xno; then
      WCSLIB_CPPFLAGS="-I$with_wcslib/include"
      acx_wcs_ldflags="-L$with_wcslib/lib"
   else
      acx_wcs_ok=disable
   fi
fi

if test $acx_wcs_ok = disable; then
   echo "**** WCSLIB explicitly disabled by configure."
else

   AC_LANG_PUSH([C])

   # Save environment

   acx_wcs_save_CPPFLAGS="$CPPFLAGS"
   acx_wcs_save_LIBS="$LIBS"

   # First, try user-specified location

   WCSLIB="$acx_wcs_ldflags $acx_wcs_libs"
   CPPFLAGS="$CPPFLAGS $WCSLIB_CPPFLAGS"
   LIBS="$WCSLIB $acx_wcs_save_LIBS $MATH"

   AC_CHECK_HEADERS([wcslib/wcs.h])
   AC_MSG_CHECKING([for wcsini in user-specified or default location])
   AC_TRY_LINK_FUNC(wcsini, [acx_wcs_ok=yes;AC_DEFINE(HAVE_WCSLIB,1,[Define if you have the wcslib library.])], [])
   AC_MSG_RESULT($acx_wcs_ok)

   if test $acx_wcs_ok = no; then
      
      # try pkg-config
      if test `pkg-config --exists wcslib`; then
         WCSLIB_CPPFLAGS=`pkg-config --cflags wcslib`
         WCSLIB=`pkg-config --libs wcslib`
         CPPFLAGS="$acx_wcs_save_CPPFLAGS $WCSLIB_CPPFLAGS"
         LIBS="$WCSLIB $acx_wcs_save_LIBS $MATH"

         AC_CHECK_HEADERS([wcslib/wcs.h])
         AC_MSG_CHECKING([for wcsini in pkg-config location])
         AC_TRY_LINK_FUNC(wcsini, [acx_wcs_ok=yes;AC_DEFINE(HAVE_WCSLIB,1,[Define if you have the wcslib library.])], [])
         AC_MSG_RESULT($acx_wcs_ok)

      fi

   fi

   if test $acx_wcs_ok = no; then
      WCSLIB_CPPFLAGS=""
      WCSLIB=""
   fi

   # Restore environment

   LIBS="$acx_wcs_save_LIBS"
   CPPFLAGS="$acx_wcs_save_CPPFLAGS"

   AC_LANG_POP([C])

fi

# Define exported variables

AC_SUBST(WCSLIB_CPPFLAGS)
AC_SUBST(WCSLIB)

# Execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:

if test x"$acx_wcs_ok" = xyes; then
   ifelse([$1],,[echo "**** Enabling wcslib support."],[$1])
else
   ifelse([$2],,[echo "**** wcslib not found - disabling support."],[$2])
fi

])
