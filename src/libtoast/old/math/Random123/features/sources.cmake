
include(MacroDefineModule)

file(GLOB TOAST_HEADERS ${CMAKE_CURRENT_LIST_DIR}/*.h ${CMAKE_CURRENT_LIST_DIR}/*.hpp)

DEFINE_MODULE(NAME libtoast.math.Random123.features
    HEADERS     ${TOAST_HEADERS}
    HEADER_EXT  ".hpp;.hh;.h"
    SOURCE_EXT  ".cpp;.cc;.c"
)

install(FILES ${TOAST_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Random123/features)

