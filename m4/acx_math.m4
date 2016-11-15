#
# SYNOPSIS
#
#   ACX_MATH([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   This macro looks for an implementation of the standard C math library.
#   Mainly this is useful for replacing the standard libm with one from a
#   different compiler vendor (like Intel's libimf).
#
#   To link an application with the math library, you should link with:
#
#   	$MATH
#
#   The user may use:
# 
#       --with-math=<flags>
#
#   to manually specify the math library linking flags
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


AC_DEFUN([ACX_MATH], [
AC_PREREQ(2.50)

acx_math_ok=no

MATH=""

AC_ARG_WITH(math, [AC_HELP_STRING([--with-math=<flags>], [use math library linking command <flags>.  Set to "no" to disable.])])

if test x"$with_math" != x; then
   if test x"$with_math" != xno; then
      MATH="$with_math"
   else
      acx_math_ok=disable
   fi
fi

if test $acx_math_ok = disable; then
   echo "**** Math library explicitly disabled by configure."
else

   # Save environment

   acx_math_save_LIBS="$LIBS"

   # Test linking

   # First, try user-specified location or linked-by-default (for example using
   # a compiler wrapper script)

   LIBS="$MATH $acx_math_save_LIBS"

   AC_CHECK_HEADERS([math.h])

   AC_MSG_CHECKING([for cos in user specified location])
   AC_TRY_LINK_FUNC(cos, [acx_math_ok=yes;AC_DEFINE(HAVE_MATH,1,[Define if you have the C math library.])], [])
   AC_MSG_RESULT($acx_math_ok)

   if test $acx_math_ok = no; then
      # try standard libm

      MATH="-lm"
      LIBS="$MATH $acx_math_save_LIBS"
	
      AC_MSG_CHECKING([for cos in default libm])
      AC_TRY_LINK_FUNC(cos, [acx_math_ok=yes;AC_DEFINE(HAVE_MATH,1,[Define if you have the C math library.])], [])
      AC_MSG_RESULT($acx_math_ok)
   fi

   # Restore environment

   LIBS="$acx_math_save_LIBS"
   CPPFLAGS="$acx_math_save_CPPFLAGS"

fi

# Define exported variables

AC_SUBST(MATH)

# Execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:

if test x"$acx_math_ok" = xyes; then
   ifelse([$1],,[echo "**** Enabling C math support."],[$1])
else
   ifelse([$2],,[echo "**** C math library not found - disabling support."],[$2])
fi

])
