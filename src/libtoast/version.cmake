
# Form the git and release version strings and update the version.cpp file if
# those versions have changed.

execute_process(
    COMMAND
    git describe --tags --always
    OUTPUT_VARIABLE GIT_DESC_RAW
    ERROR_QUIET
)
string(STRIP "${GIT_DESC_RAW}" GIT_DESC)

string(REGEX REPLACE "^([0-9]+\\.[0-9]+\\.[0-9a-z]+).*" "\\1" VERSION_TAG "${GIT_DESC}")
string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.[0-9a-z]+-([0-9]+)-.*" "\\1" DEV_COUNT "${GIT_DESC}")

if ("${DEV_COUNT}" STREQUAL "${GIT_DESC}")
    set(DEV_COUNT "0")
endif()

# Check whether we got any revision (if this is a git checkout).  Form a
# PEP compatible version string.

if ("${GIT_DESC}" STREQUAL "")
    set(GIT_VERSION "")
else()
    if ("${DEV_COUNT}" STREQUAL "0")
        # We are on a tag
        set(GIT_VERSION "${VERSION_TAG}")
    else()
        set(GIT_VERSION "${VERSION_TAG}.dev${DEV_COUNT}")
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
