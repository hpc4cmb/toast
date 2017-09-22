#
# SYNOPSIS
#
#   ACX_TBB([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   This macro tries to link against the Intel Thread Building Block library.
#
#   To link an application with TBB, you should link with:
#
#   	$TBB
#
#   The user may use:
# 
#       --with-tbb=<dir>
#
#   to manually specify the directory containing the architecture specific
#   library files.
#
#   ACTION-IF-FOUND is a list of shell commands to run if the TBB library is
#   found, and ACTION-IF-NOT-FOUND is a list of commands to run it if it is
#   not found. If ACTION-IF-FOUND is not specified, the default action will
#   define HAVE_TBB and set output variables above.
#
#   This macro requires autoconf 2.50 or later.
#
# LAST MODIFICATION
#
#   2017-09-19
#
# COPYING
#
#   Copyright (c) 2017 Theodore Kisner <tskisner@lbl.gov>
#
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   o  Redistributions of source code must retain the above copyright notice, 
#      this list of conditions and the following disclaimer.
#
#   o  Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
#   PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


AC_DEFUN([ACX_TBB], [
AC_PREREQ(2.50)
AC_REQUIRE([AX_PTHREAD])

acx_tbb_ok=no
acx_tbb_ldflags=""

TBB_CPPFLAGS=""
TBB=""

AC_ARG_WITH(tbb, [AC_HELP_STRING([--with-tbb=<dir>], [use TBB architecture-specific libraries installed in <dir>.  Set to "no" to disable.])])

if test x"$with_tbb" != x; then
   if test x"$with_tbb" != xno; then
      TBB_CPPFLAGS="-I$with_tbb/../../../include"
      acx_tbb_ldflags="-L$with_tbb"
   else
      acx_tbb_ok=disable
   fi
fi

if test $acx_tbb_ok = disable; then
   echo "**** TBB explicitly disabled by configure."
else

   # Save environment

   acx_tbb_save_CPPFLAGS="$CPPFLAGS"
   acx_tbb_save_LIBS="$LIBS"

   # Test compile and linking.

   # First, try user-specified location or linked-by-default (for example using
   # a compiler wrapper script)

   TBB="$acx_tbb_ldflags -ltbb -ltbbmalloc"
   CPPFLAGS="$acx_tbb_save_CPPFLAGS $PTHREAD_CFLAGS $TBB_CPPFLAGS"
   LIBS="$acx_tbb_save_LIBS $TBB $PTHREAD_LIBS"

   AC_CHECK_HEADERS([tbb/tbb.h])

   AC_MSG_CHECKING([for parallel_for])

   AC_LINK_IFELSE([AC_LANG_PROGRAM([
      [#include <tbb/task_scheduler_init.h>]
   ],[
      tbb::task_scheduler_init init;
   ])],[acx_tbb_ok=yes;AC_DEFINE(HAVE_TBB,1,[Define if you have the Thread Building Blocks library.])])

   AC_MSG_RESULT($acx_tbb_ok)

   if test $acx_tbb_ok = no; then
      TBB_CPPFLAGS=""
      TBB=""
   fi

   # Restore environment

   LIBS="$acx_tbb_save_LIBS"
   CPPFLAGS="$acx_tbb_save_CPPFLAGS"

fi

# Define exported variables

AC_SUBST(TBB_CPPFLAGS)
AC_SUBST(TBB)

# Execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:

if test x"$acx_tbb_ok" = xyes; then
   ifelse([$1],,[echo "**** Enabling TBB support."],[$1])
else
   ifelse([$2],,[echo "**** TBB not found - disabling support."],[$2])
fi

])
