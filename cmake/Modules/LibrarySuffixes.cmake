#
#
#

if(__librarysuffixes_isloaded)
    return()
endif(__librarysuffixes_isloaded)
set(__librarysuffixes_isloaded ON)

STRING(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type)

set(DEFAULT_SHARED_LIBRARY_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX}
    CACHE STRING "Default shared library suffix")
set(DEFAULT_STATIC_LIBRARY_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX}
    CACHE STRING "Default static library suffix")
set(DEFAULT_EXECUTABLE_SUFFIX ${CMAKE_EXECUTABLE_SUFFIX}
    CACHE STRING "Default executable suffix")

if("${_build_type}" STREQUAL "DEBUG")

    set(CMAKE_SHARED_LIBRARY_SUFFIX "-debug${DEFAULT_SHARED_LIBRARY_SUFFIX}")
    set(CMAKE_STATIC_LIBRARY_SUFFIX "-debug${DEFAULT_STATIC_LIBRARY_SUFFIX}")
    #set(CMAKE_EXECUTABLE_SUFFIX     "-debug${DEFAULT_EXECUTABLE_SUFFIX}")

elseif("${_build_type}" STREQUAL "RELWITHDEBINFO")

    set(CMAKE_SHARED_LIBRARY_SUFFIX "-rdebug${DEFAULT_SHARED_LIBRARY_SUFFIX}")
    set(CMAKE_STATIC_LIBRARY_SUFFIX "-rdebug${DEFAULT_STATIC_LIBRARY_SUFFIX}")
    #set(CMAKE_EXECUTABLE_SUFFIX     "-rdebug${DEFAULT_EXECUTABLE_SUFFIX}")

elseif("${_build_type}" STREQUAL "MINSIZEREL")

    set(CMAKE_SHARED_LIBRARY_SUFFIX "-minsize${DEFAULT_SHARED_LIBRARY_SUFFIX}")
    set(CMAKE_STATIC_LIBRARY_SUFFIX "-minsize${DEFAULT_STATIC_LIBRARY_SUFFIX}")
    #set(CMAKE_EXECUTABLE_SUFFIX     "-minsize${DEFAULT_EXECUTABLE_SUFFIX}")

endif()

unset(_build_type)
mark_as_advanced(DEFAULT_SHARED_LIBRARY_SUFFIX
    DEFAULT_STATIC_LIBRARY_SUFFIX
    DEFAULT_EXECUTABLE_SUFFIX)
