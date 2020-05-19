
# Form the git and release version strings and update the version.cpp file if
# those versions have changed.

execute_process(
    COMMAND
    git describe --tags --dirty --always
    OUTPUT_VARIABLE GIT_DESC_RAW
    ERROR_QUIET
)
string(STRIP "${GIT_DESC_RAW}" GIT_DESC)
string(REGEX REPLACE "-.*$" "" GIT_LAST "${GIT_DESC}")

execute_process(
    COMMAND
    git rev-list --count HEAD
    OUTPUT_VARIABLE GIT_COUNT
    ERROR_QUIET
)
string(STRIP "${GIT_COUNT}" GIT_DEV)

# Check whether we got any revision (if this is a git checkout).  Form a
# PEP compatible version string.

if ("${GIT_LAST}" STREQUAL "")
    set(GIT_VERSION "")
else()
    if ("${GIT_DEV}" STREQUAL "0")
        # We are on a tag
        set(GIT_VERSION "${GIT_LAST}")
    else()
        set(GIT_VERSION "${GIT_LAST}.dev${GIT_DEV}")
    endif()
endif()

# Separately parse the special "RELEASE" file that is manually updated when
# making an actual release.

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../toast/RELEASE)
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/../toast/RELEASE REL_VERSION)
    string(STRIP "${REL_VERSION}" RELEASE_VERSION)
else()
    set(RELEASE_VERSION "")
endif()

set(VERSION "const char* GIT_VERSION = \"${GIT_VERSION}\";
const char* RELEASE_VERSION = \"${RELEASE_VERSION}\";")

if(EXISTS version.cpp)
    file(READ version.cpp VERSION_)
    string(STRIP "${VERSION_}" CHECK_VERSION)
else()
    set(CHECK_VERSION "")
endif()

if (NOT "${VERSION}" STREQUAL "${CHECK_VERSION}")
    file(WRITE version.cpp "${VERSION}")
endif()
