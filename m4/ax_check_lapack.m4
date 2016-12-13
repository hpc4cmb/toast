# SYNOPSIS
#
#   AX_CHECK_LAPACK([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   This macro looks for a library that implements the LAPACK linear-algebra
#   interface (see http://www.netlib.org/lapack/). On success, it sets the
#   LAPACK_LIBS output variable to hold the requisite library linkages.
#
#   To link with LAPACK, you should link with:
#
#       $LAPACK_LIBS $BLAS_LIBS $LIBS
#
#   in that order. BLAS_LIBS is the output variable of the AX_BLAS macro,
#   called automatically. 
#
#   The user may also use --with-lapack=<lib> in order to use some specific
#   LAPACK library <lib>.
#
#   ACTION-IF-FOUND is a list of shell commands to run if a LAPACK library
#   is found, and ACTION-IF-NOT-FOUND is a list of commands to run it if it
#   is not found. If ACTION-IF-FOUND is not specified, the default action
#   will define HAVE_LAPACK.
#
#   The macro is a modified version of ACX_LAPACK, from the autoconf macro
#   archive.  The original macro depends on a Fortran compiler, and this
#   version does not.  This version has not been tested extensively, but
#   it is known to work with ATLAS and vecLib.
#
# LAST MODIFICATION
#
#   2014-01-08
#
# COPYLEFT
#
#   Copyright (c) 2014 Theodore Kisner <tskisner.public@gmail.com>
#   Copyright (c) 2008 Patrick O. Perry  <patperry@stanfordalumni.org>
#   Copyright (c) 2008 Steven G. Johnson <stevenj@alum.mit.edu>
#
#   This program is free software: you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation, either version 3 of the License, or (at your
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#   Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program. If not, see <http://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright owner
#   gives unlimited permission to copy, distribute and modify the configure
#   scripts that are the output of Autoconf when processing the Macro. You
#   need not follow the terms of the GNU General Public License when using
#   or distributing such scripts, even though portions of the text of the
#   Macro appear in them. The GNU General Public License (GPL) does govern
#   all other use of the material that constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the Autoconf
#   Macro released by the Autoconf Macro Archive. When you make and
#   distribute a modified version of the Autoconf Macro, you may extend this
#   special exception to the GPL to apply to your modified version as well.

AC_DEFUN([AX_CHECK_LAPACK], [
AC_REQUIRE([AX_CHECK_BLAS])
ax_lapack_ok=no

LAPACK_LIBS=""

AC_ARG_WITH([lapack], [AC_HELP_STRING([--with-lapack=<link command>], [use specified LAPACK link command.  Set to "no" to disable.])])

if test x"$with_lapack" != x; then
   if test x"$with_lapack" != xno; then
      LAPACK_LIBS="$with_lapack"
   else
      ax_lapack_ok=disable
   fi
fi

if test $ax_lapack_ok = disable; then
   echo "**** LAPACK explicitly disabled by configure."
else

    # Get fortran linker name of LAPACK function to check for.
    if test x"$ax_blas_underscore" = xyes; then
        cheev=cheev_
    else
        cheev=cheev
    fi

    # We cannot use LAPACK if BLAS is not found
    if test "x$ax_blas_ok" != xyes; then
        ax_lapack_ok=noblas
        LAPACK_LIBS=""
    fi

    # First, check LAPACK_LIBS environment variable
    if test "x$LAPACK_LIBS" != x; then
        save_LIBS="$LIBS"; LIBS="$LAPACK_LIBS $BLAS_LIBS $LIBS"
        AC_MSG_CHECKING([for $cheev in $LAPACK_LIBS])
        AC_TRY_LINK_FUNC($cheev, [ax_lapack_ok=yes], [LAPACK_LIBS=""])
        AC_MSG_RESULT($ax_lapack_ok)
        LIBS="$save_LIBS"
        if test $ax_lapack_ok = no; then
            LAPACK_LIBS=""
        fi
    fi

    # LAPACK linked to by default?  (is sometimes included in BLAS lib)
    if test $ax_lapack_ok = no; then
        save_LIBS="$LIBS"; LIBS="$BLAS_LIBS $LIBS"
        AC_CHECK_FUNC($cheev, [ax_lapack_ok=yes])
        LIBS="$save_LIBS"
    fi

    # Generic LAPACK library?
    for lapack in lapack lapack_rs6k; do
        if test $ax_lapack_ok = no; then
            save_LIBS="$LIBS"; LIBS="$BLAS_LIBS $LIBS"
            AC_CHECK_LIB($lapack, $cheev,
                [ax_lapack_ok=yes; LAPACK_LIBS="-l$lapack"], [], [])
            LIBS="$save_LIBS"
        fi
    done

fi

AC_SUBST(LAPACK_LIBS)

# Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
if test x"$ax_lapack_ok" = xyes; then
        ifelse([$1],,AC_DEFINE(HAVE_LAPACK,1,[Define if you have LAPACK library.]),[$1])
        :
else
        ax_lapack_ok=no
        $2
fi
])dnl AX_LAPACK
