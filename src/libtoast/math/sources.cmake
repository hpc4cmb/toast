
include(MacroDefineModule)

file(GLOB TOAST_HEADERS ${CMAKE_CURRENT_LIST_DIR}/toast/*.hpp)
file(GLOB TOAST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/toast/*.cpp)

set(FLAGS "-fopt-info-vec-optimized -fopt-info-vec-missed")
if(CMAKE_CXX_COMPILER_IS_INTEL)
    set(FLAGS "-qopt-report -qopt-report-phase=vec -diag-enable=vec")
endif()
                        
DEFINE_MODULE(NAME libtoast.math
    HEADERS     ${TOAST_HEADERS}
    SOURCES     ${TOAST_SOURCES}
    HEADER_EXT  ".hpp;.hh;.h"
    SOURCE_EXT  ".cpp;.cc;.c"
)

install(FILES ${TOAST_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/toast)
