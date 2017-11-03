
include(MacroDefineModule)

file(GLOB TOAST_HEADERS ${CMAKE_CURRENT_LIST_DIR}/toast/*.hpp)
file(GLOB TOAST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/toast/*.cpp)

DEFINE_MODULE(NAME libtoast.test
    EXCLUDE     data_healpix
    HEADERS     ${TOAST_HEADERS}
    SOURCES     ${TOAST_SOURCES}
    HEADER_EXT  ".hpp;.hh;.h"
    SOURCE_EXT  ".cpp;.cc;.c"
)

