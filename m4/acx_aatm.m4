#
# SYNOPSIS
#
#   ACX_AATM([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   This macro looks for a version of the aatm library.  The AATM_CPPFLAGS
#   and AATM output variables hold the compile and link flags.
#
#   To link an application with aatm, you should link with:
#
#   	$AATM
#
#   The user may use:
# 
#       --with-aatm=<PATH>
#
#   to manually specify the aatm installation prefix.
#
#   ACTION-IF-FOUND is a list of shell commands to run if an aatm library is
#   found, and ACTION-IF-NOT-FOUND is a list of commands to run it if it is
#   not found. If ACTION-IF-FOUND is not specified, the default action will
#   define HAVE_AATM and set output variables above.
#
#   This macro requires autoconf 2.50 or later.
#
# LAST MODIFICATION
#
#   2017-12-21
#
# COPYING
#
#   Copyright (c) 2017 Reijo Keskitalo <rtkeskitalo@lbl.gov>
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

AC_DEFUN([ACX_AATM], [
AC_PREREQ(2.50)
AC_REQUIRE([AX_PROG_CXX_MPI])

acx_aatm_ok=no

acx_aatm_default="-laatm"

AATM_CPPFLAGS=""
AATM_LDFLAGS=""
AATM_LIBS="$acx_aatm_default"

AC_ARG_WITH(aatm, [AC_HELP_STRING([--with-aatm=<PATH>], [use the aatm installed in <PATH>.])])

if test x"$with_aatm" != x; then
   if test x"$with_aatm" != xno; then
      AATM_CPPFLAGS="-I$with_aatm/include"
      AATM_LDFLAGS="-L$with_aatm/lib"
   else
      acx_aatm_ok=disable
   fi
fi

if test $acx_aatm_ok = disable; then
   echo "**** aatm explicitly disabled by configure."
else

   # Save environment

   acx_aatm_save_CPPFLAGS="$CPPFLAGS"
   acx_aatm_save_LIBS="$LIBS"

   # Test MPI compile and linking.  We need the eigensolver capabilities,
   # so check that the library has been built with eigen routines.

   CPPFLAGS="$CPPFLAGS $AATM_CPPFLAGS"
   AATM="$AATM_LDFLAGS $AATM_LIBS"
   LIBS="$AATM $acx_aatm_save_LIBS $LIBS $FCLIBS -lm"

   AC_CHECK_HEADERS([ATMPressure.h])

   AC_MSG_CHECKING([for atm::Pressure in $AATM])
   AC_LINK_IFELSE([AC_LANG_PROGRAM([
      [#include "ATMPressure.h"]
   ],[
      using namespace std;
      using namespace atm;
      Pressure p(1000, "mb");
   ])],[acx_aatm_ok=yes;AC_DEFINE(HAVE_AATM,1,[Define if you have the aatm library.])])

   AC_MSG_RESULT($acx_aatm_ok)

   if test $acx_aatm_ok = no; then
      AATM=""
   fi

   # Restore environment

   CPPFLAGS="$acx_aatm_save_CPPFLAGS"
   LIBS="$acx_aatm_save_LIBS"

fi

# Define exported variables

AC_SUBST(AATM_CPPFLAGS)
AC_SUBST(AATM)

# Execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
   
if test x"$acx_aatm_ok" = xyes; then
   ifelse([$1],,[echo "**** Enabling support for aatm."],[$1])
else
   ifelse([$2],,[echo "**** aatm not found - disabling support."],[$2])
fi

])
