#
# Define GNU standard installation directories
#
# Provides install directory variables as defined by the
# `GNU Coding Standards`_.
#
# .. _`GNU Coding Standards`: https://www.gnu.org/prep/standards/html_node/Directory-Variables.html
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# Inclusion of this module defines the following variables:
#
#   CMAKE_INSTALL_<dir>
#
#   Destination for files of a given type.  This value may be passed to
#   the   DESTINATION   options of :command:`install` commands for the
#   corresponding file type.
#
#   CMAKE_INSTALL_FULL_<dir>
#
#   The absolute path generated from the corresponding   CMAKE_INSTALL_<dir>
#   value.  If the value is not already an absolute path, an absolute path
#   is constructed typically by prepending the value of the
#   :variable:`CMAKE_INSTALL_PREFIX` variable.  However, there are some
#   `special cases`_ as documented below.
#
# where   <dir>   is one of:
#
#   BINDIR
#       user executables (bin)
#   SBINDIR
#       system admin executables (sbin)
#   LIBEXECDIR
#       program executables (libexec)
#   SYSCONFDIR
#       read-only single-machine data (etc)
#   SHAREDSTATEDIR
#       modifiable architecture-independent data (com)
#   LOCALSTATEDIR
#       modifiable single-machine data (var)
#   LIBDIR
#       object code libraries (lib or lib64 or lib/<multiarch-tuple> on Debian)
#   INCLUDEDIR
#       C header files (include)
#   OLDINCLUDEDIR
#       C header files for non-gcc (/usr/include)
#   DATAROOTDIR
#       read-only architecture-independent data root (share)
#   DATADIR
#       read-only architecture-independent data (DATAROOTDIR)
#   INFODIR
#       info documentation (DATAROOTDIR/info)
#   LOCALEDIR
#       locale-dependent data (DATAROOTDIR/locale)
#   MANDIR
#       man documentation (DATAROOTDIR/man)
#   DOCDIR
#       documentation root (DATAROOTDIR/doc/PROJECT_NAME)
#
# ---------------------------------------------------------------------------- #
#   Use default GNU install directories script
include(GNUInstallDirs)

# ---------------------------------------------------------------------------- #
#   Store the _LIBDIR_DEFAULT variable
set(CMAKE_INSTALL_LIBDIR_DEFAULT ${_LIBDIR_DEFAULT}
    CACHE PATH "CMake library directory path suffix")

# ---------------------------------------------------------------------------- #
#   Add one for cmake export
set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR_DEFAULT}/cmake/${CMAKE_PROJECT_NAME}"
    CACHE PATH "CMake export directory")
unset(_LIBDIR_DEFAULT)

# ---------------------------------------------------------------------------- #
#   Add one for cmake export
if(NOT "${PYTHON_VERSION_STRING}" STREQUAL "")
    GET_VERSION_COMPONENTS(PYTHON "${PYTHON_VERSION_STRING}")
endif(NOT "${PYTHON_VERSION_STRING}" STREQUAL "")
set(CMAKE_INSTALL_PYTHONDIR
    "${CMAKE_INSTALL_LIBDIR_DEFAULT}/python${PYTHON_SHORT_VERSION}/site-packages"
    CACHE PATH "CMake python install directory" FORCE)

# ---------------------------------------------------------------------------- #
#   Add the cmake full directory variable
#
foreach(TYPE CMAKE PYTHON)
    if(NOT IS_ABSOLUTE ${CMAKE_INSTALL_${TYPE}DIR})
        set(CMAKE_INSTALL_FULL_${TYPE}DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_${TYPE}DIR}")
    else()
        set(CMAKE_INSTALL_FULL_${TYPE}DIR "${CMAKE_INSTALL_${TYPE}DIR}")
        set(CMAKE_INSTALL_IS_NONRELOCATABLE 1)
    endif()
endforeach()

# ---------------------------------------------------------------------------- #
#   Function for getting a relative path to an installation directory
#
#   INSTALL_RELATIVE_PATH(<VAR> <TYPE> <REFERENCE_FOLDER>)
#       VAR:               variable name
#       TYPE:              install directory type (e.g. BIN, INCLUDE, LIB, etc.)
#       REFERENCE_FOLDER:  relative to this directory
#
#   example: getting relative path of include directory from cmake directory
#       INSTALL_RELATIVE_PATH(cmake_incdir INCLUDE ${CMAKE_INSTALL_FULL_CMAKEDIR})
#
function(INSTALL_RELATIVE_PATH VAR TYPE REFERENCE_FOLDER)
    file(RELATIVE_PATH _tmp${VAR} CMAKE_INSTALL_FULL_${TYPE}DIR ${REFERENCE_FOLDER})
    set(${VAR} ${_tmp${VAR}} PARENT_SCOPE)
endfunction(INSTALL_RELATIVE_PATH TYPE REFERENCE_FOLDER)

