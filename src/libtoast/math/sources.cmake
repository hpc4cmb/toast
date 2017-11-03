
include(MacroDefineModule)

file(GLOB TOAST_HEADERS ${CMAKE_CURRENT_LIST_DIR}/toast/*.hpp)
file(GLOB TOAST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/toast/*.cpp)

#set(RANDOM_DIR ${CMAKE_CURRENT_LIST_DIR}/Random123)
#file(GLOB RANDOM_HEADERS_A ${RANDOM_DIR}/*.h ${RANDOM_DIR}/.hpp)
#file(GLOB RANDOM_HEADERS_C ${RANDOM_DIR}/conventional/*.h ${RANDOM_DIR}/conventional/*.hpp)
#file(GLOB RANDOM_HEADERS_F ${RANDOM_DIR}/features/)

set(FLAGS "-fopt-info-vec-optimized -fopt-info-vec-missed")
if(CMAKE_CXX_COMPILER_IS_INTEL)
    set(FLAGS "-qopt-report -qopt-report-phase=vec -diag-enable=vec")
endif()

#add_compile_flags(SOURCES
#    toast_qarray.cpp
#    toast_rng.cpp
#    COMPILE_FLAGS "${FLAGS}")
                        
DEFINE_MODULE(NAME libtoast.math
    HEADERS     ${TOAST_HEADERS}
    SOURCES     ${TOAST_SOURCES}
    HEADER_EXT  ".hpp;.hh;.h"
    SOURCE_EXT  ".cpp;.cc;.c"
)

install(FILES ${TOAST_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/toast)
