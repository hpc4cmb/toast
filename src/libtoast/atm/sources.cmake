
include(MacroDefineModule)

if(USE_ELEMENTAL)

    file(GLOB TOAST_HEADERS ${CMAKE_CURRENT_LIST_DIR}/toast/*.hpp)
    file(GLOB TOAST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/toast/*.cpp)

    DEFINE_MODULE(NAME libtoast.atm
        HEADERS     ${TOAST_HEADERS}
        SOURCES     ${TOAST_SOURCES}
        HEADER_EXT  ".hpp;.hh;.h"
        SOURCE_EXT  ".cpp;.cc;.c"
    )

    install(FILES ${TOAST_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/toast)

endif(USE_ELEMENTAL)

