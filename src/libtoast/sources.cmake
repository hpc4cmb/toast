
include(MacroDefineModule)

DEFINE_MODULE(NAME libtoast.main
    HEADERS     ${CMAKE_BINARY_DIR}/config/config.h
    EXCLUDE     toast_test
    HEADER_EXT  .hpp .hh .h
    SOURCE_EXT  .cpp .cc .c
)

install(FILES ${CMAKE_CURRENT_LIST_DIR}/toast.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
