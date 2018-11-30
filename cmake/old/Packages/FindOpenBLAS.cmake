
foreach (OpenBLAS_NAME openblaso openblasp openblas)
    FIND_LIBRARY(${OpenBLAS_NAME}_LIBRARY
        NAMES ${OpenBLAS_NAME}

        HINTS
        ${OpenBLAS_ROOT}
        $ENV{OpenBLAS_ROOT}
        
        PATH_SUFFIXES
        ${CMAKE_INSTALL_LIBDIR_DEFAULT}
        lib64
        lib
        
        PATHS 
        /usr/lib64
        /usr/lib
        /usr/local/lib64
        /usr/local/lib
    )
  
    # over-ride with current found library
    set(OpenBLAS_LIBRARY ${${OpenBLAS_NAME}_LIBRARY})
    
endforeach()


# use only one library
if(OpenBLAS_TMP_LIBRARIES)
  list(GET OpenBLAS_TMP_LIBRARIES 0 OpenBLAS_LIBRARY)
endif()


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS DEFAULT_MSG OpenBLAS_LIBRARY)

set(OpenBLAS_LIBRARIES ${OpenBLAS_LIBRARY})

