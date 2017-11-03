
include(MacroDefineModule)

file(GLOB TOAST_HEADERS ${CMAKE_CURRENT_LIST_DIR}/toast/*.hpp)
file(GLOB TOAST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/toast/*.cpp)

#set(RANDOM_DIR ${CMAKE_CURRENT_LIST_DIR}/Random123)
#file(GLOB RANDOM_HEADERS_A ${RANDOM_DIR}/*.h ${RANDOM_DIR}/.hpp)
#file(GLOB RANDOM_HEADERS_C ${RANDOM_DIR}/conventional/*.h ${RANDOM_DIR}/conventional/*.hpp)
#file(GLOB RANDOM_HEADERS_F ${RANDOM_DIR}/features/)

DEFINE_MODULE(NAME libtoast.math
    HEADERS     ${TOAST_HEADERS}
    SOURCES     ${TOAST_SOURCES}
    HEADER_EXT  ".hpp;.hh;.h"
    SOURCE_EXT  ".cpp;.cc;.c"
)

install(FILES ${TOAST_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/toast)
