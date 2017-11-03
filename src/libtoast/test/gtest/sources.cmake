
include(MacroDefineModule)

file(GLOB GTEST_HEADERS ${CMAKE_CURRENT_LIST_DIR}/gtest/*.h
    ${CMAKE_CURRENT_LIST_DIR}/gtest/internal/*.h
    ${CMAKE_CURRENT_LIST_DIR}/gtest/internal/custom/*.h)

DEFINE_MODULE(NAME libtoast.test.gtest
    HEADERS     ${GTEST_HEADERS}
    SOURCES     ${CMAKE_CURRENT_LIST_DIR}/gtest.cc
                ${CMAKE_CURRENT_LIST_DIR}/gtest-port.cc
                ${CMAKE_CURRENT_LIST_DIR}/gtest-death-test.cc
                ${CMAKE_CURRENT_LIST_DIR}/gtest-test-part.cc
                ${CMAKE_CURRENT_LIST_DIR}/gtest-printers.cc
                ${CMAKE_CURRENT_LIST_DIR}/gtest-filepath.cc
    HEADER_EXT  .hpp .hh .h
)

