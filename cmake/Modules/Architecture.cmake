

macro(AutodetectHostArchitecture)
    set(TARGET_ARCHITECTURE "generic")
    set(Vc_ARCHITECTURE_FLAGS)
    set(_vendor_id)
    set(_cpu_family)
    set(_cpu_model)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        file(READ "/proc/cpuinfo" _cpuinfo)
        string(REGEX REPLACE ".*vendor_id[ \t]*:[ \t]+([a-zA-Z0-9_-]+).*" "\\1" _vendor_id "${_cpuinfo}")
        string(REGEX REPLACE ".*cpu family[ \t]*:[ \t]+([a-zA-Z0-9_-]+).*" "\\1" _cpu_family "${_cpuinfo}")
        string(REGEX REPLACE ".*model[ \t]*:[ \t]+([a-zA-Z0-9_-]+).*" "\\1" _cpu_model "${_cpuinfo}")
        string(REGEX REPLACE ".*flags[ \t]*:[ \t]+([^\n]+).*" "\\1" _cpu_flags "${_cpuinfo}")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        exec_program("/usr/sbin/sysctl -n machdep.cpu.vendor" OUTPUT_VARIABLE _vendor_id)
        exec_program("/usr/sbin/sysctl -n machdep.cpu.model"  OUTPUT_VARIABLE _cpu_model)
        exec_program("/usr/sbin/sysctl -n machdep.cpu.family" OUTPUT_VARIABLE _cpu_family)
        exec_program("/usr/sbin/sysctl -n machdep.cpu.features" OUTPUT_VARIABLE _cpu_flags)
        string(TOLOWER "${_cpu_flags}" _cpu_flags)
        string(REPLACE "." "_" _cpu_flags "${_cpu_flags}")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        get_filename_component(_vendor_id "[HKEY_LOCAL_MACHINE\\Hardware\\Description\\System\\CentralProcessor\\0;VendorIdentifier]" NAME CACHE)
        get_filename_component(_cpu_id "[HKEY_LOCAL_MACHINE\\Hardware\\Description\\System\\CentralProcessor\\0;Identifier]" NAME CACHE)
        mark_as_advanced(_vendor_id _cpu_id)
        string(REGEX REPLACE ".* Family ([0-9]+) .*" "\\1" _cpu_family "${_cpu_id}")
        string(REGEX REPLACE ".* Model ([0-9]+) .*" "\\1" _cpu_model "${_cpu_id}")
    endif(CMAKE_SYSTEM_NAME STREQUAL "Linux")

    if(_vendor_id STREQUAL "GenuineIntel")
        if(_cpu_family EQUAL 6)
            # taken from the Intel ORM
            # http://www.intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html
            # CPUID Signature Values of Of Recent Intel Microarchitectures
            # 4E 5E       | Skylake microarchitecture
            # 3D 47 56    | Broadwell microarchitecture
            # 3C 45 46 3F | Haswell microarchitecture
            # 3A 3E       | Ivy Bridge microarchitecture
            # 2A 2D       | Sandy Bridge microarchitecture
            # 25 2C 2F    | Intel microarchitecture Westmere
            # 1A 1E 1F 2E | Intel microarchitecture Nehalem
            # 17 1D       | Enhanced Intel Core microarchitecture
            # 0F          | Intel Core microarchitecture
            #
            # Intel SDM Vol. 3C 35-1 / December 2016:
            # 57          | Xeon Phi 3200, 5200, 7200  [Knights Landing]
            # 85          | Future Xeon Phi
            # 8E 9E       | 7th gen. Core              [Kaby Lake]
            # 55          | Future Xeon                [Skylake w/ AVX512]
            # 4E 5E       | 6th gen. Core / E3 v5      [Skylake w/o AVX512]
            # 56          | Xeon D-1500                [Broadwell]
            # 4F          | Xeon E5 v4, E7 v4, i7-69xx [Broadwell]
            # 47          | 5th gen. Core / Xeon E3 v4 [Broadwell]
            # 3D          | M-5xxx / 5th gen.          [Broadwell]
            # 3F          | Xeon E5 v3, E7 v3, i7-59xx [Haswell-E]
            # 3C 45 46    | 4th gen. Core, Xeon E3 v3  [Haswell]
            # 3E          | Xeon E5 v2, E7 v2, i7-49xx [Ivy Bridge-E]
            # 3A          | 3rd gen. Core, Xeon E3 v2  [Ivy Bridge]
            # 2D          | Xeon E5, i7-39xx           [Sandy Bridge]
            # 2F          | Xeon E7
            # 2A          | Xeon E3, 2nd gen. Core     [Sandy Bridge]
            # 2E          | Xeon 7500, 6500 series
            # 25 2C       | Xeon 3600, 5600 series, Core i7, i5 and i3
            #
            # Values from the Intel SDE:
            # 5C | Goldmont
            # 5A | Silvermont
            # 57 | Knights Landing
            # 66 | Cannonlake
            # 55 | Skylake Server
            # 4E | Skylake Client
            # 3C | Broadwell (likely a bug in the SDE)
            # 3C | Haswell

            macro(issue_message _family _model _use)
                set(_msg "Your CPU (family ${_family}, model ${_model}) is not known.")
                set(_msg "${_msg} Auto-detection of optimization flags failed and will")
                set(_msg "${_msg} use the ${_use} settings.")
                message(WARNING "${_msg}")
                unset(_msg)
            endmacro()

            if(_cpu_model EQUAL 87) # 57
                set(TARGET_ARCHITECTURE "knl")  # Knights Landing
            elseif(_cpu_model EQUAL 92)
                set(TARGET_ARCHITECTURE "goldmont")
            elseif(_cpu_model EQUAL 90 OR _cpu_model EQUAL 76)
                set(TARGET_ARCHITECTURE "silvermont")
            elseif(_cpu_model EQUAL 102)
                set(TARGET_ARCHITECTURE "cannonlake")
            elseif(_cpu_model EQUAL 142 OR _cpu_model EQUAL 158) # 8E, 9E
                set(TARGET_ARCHITECTURE "kaby-lake")
            elseif(_cpu_model EQUAL 85) # 55
                set(TARGET_ARCHITECTURE "skylake-avx512")
            elseif(_cpu_model EQUAL 78 OR _cpu_model EQUAL 94) # 4E, 5E
                set(TARGET_ARCHITECTURE "skylake")
            elseif(_cpu_model EQUAL 61 OR _cpu_model EQUAL 71 OR
                    _cpu_model EQUAL 79 OR _cpu_model EQUAL 86) # 3D, 47, 4F, 56
                set(TARGET_ARCHITECTURE "broadwell")
            elseif(_cpu_model EQUAL 60 OR _cpu_model EQUAL 69 OR
                    _cpu_model EQUAL 70 OR _cpu_model EQUAL 63)
                set(TARGET_ARCHITECTURE "haswell")
            elseif(_cpu_model EQUAL 58 OR _cpu_model EQUAL 62)
                set(TARGET_ARCHITECTURE "ivy-bridge")
            elseif(_cpu_model EQUAL 42 OR _cpu_model EQUAL 45)
                set(TARGET_ARCHITECTURE "sandy-bridge")
            elseif(_cpu_model EQUAL 37 OR _cpu_model EQUAL 44 OR
                    _cpu_model EQUAL 47)
                set(TARGET_ARCHITECTURE "westmere")
            elseif(_cpu_model EQUAL 26 OR _cpu_model EQUAL 30 OR
                    _cpu_model EQUAL 31 OR _cpu_model EQUAL 46)
                set(TARGET_ARCHITECTURE "nehalem")
            elseif(_cpu_model EQUAL 23 OR _cpu_model EQUAL 29)
                set(TARGET_ARCHITECTURE "penryn")
            elseif(_cpu_model EQUAL 15)
                set(TARGET_ARCHITECTURE "merom")
            elseif(_cpu_model EQUAL 28)
                set(TARGET_ARCHITECTURE "atom")
            elseif(_cpu_model EQUAL 14)
                set(TARGET_ARCHITECTURE "core")
            elseif(_cpu_model LESS 14)
                issue_message(${_cpu_family} ${_cpu_model} "generic CPU w/ SSE2")
                set(TARGET_ARCHITECTURE "generic")
            else()
                message(WARNING )
                issue_message(${_cpu_family} ${_cpu_model} "65nm Core 2 CPU")
                set(TARGET_ARCHITECTURE "merom")
            endif()
        elseif(_cpu_family EQUAL 7) # Itanium (not supported)
            message(WARNING
                "Your CPU (Itanium: family ${_cpu_family}, model ${_cpu_model}) is not supported.")
        elseif(_cpu_family EQUAL 15) # NetBurst
            list(APPEND _available_vector_units_list "sse" "sse2")
            if(_cpu_model GREATER 2) # Not sure whether this must be 3 or even 4 instead
                list(APPEND _available_vector_units_list "sse" "sse2" "sse3")
            endif(_cpu_model GREATER 2)
        endif(_cpu_family EQUAL 6)
    elseif(_vendor_id STREQUAL "AuthenticAMD")
        if(_cpu_family EQUAL 23)
            set(TARGET_ARCHITECTURE "zen")
        elseif(_cpu_family EQUAL 22) # 16h
            set(TARGET_ARCHITECTURE "AMD 16h")
        elseif(_cpu_family EQUAL 21) # 15h
            if(_cpu_model LESS 2)
                set(TARGET_ARCHITECTURE "bulldozer")
            else()
                set(TARGET_ARCHITECTURE "piledriver")
            endif()
        elseif(_cpu_family EQUAL 20) # 14h
            set(TARGET_ARCHITECTURE "AMD 14h")
        elseif(_cpu_family EQUAL 18) # 12h
        elseif(_cpu_family EQUAL 16) # 10h
            set(TARGET_ARCHITECTURE "barcelona")
        elseif(_cpu_family EQUAL 15)
            set(TARGET_ARCHITECTURE "k8")
            if(_cpu_model GREATER 64)
                set(TARGET_ARCHITECTURE "k8-sse3")
            endif(_cpu_model GREATER 64)
        endif()
    endif(_vendor_id STREQUAL "GenuineIntel")
endmacro()

# =============================================================================#

function(ArchitectureFlags VAR)

    set(TARGET_ARCHITECTURE "auto" CACHE STRING "Target architecture")

    if(TARGET_ARCHITECTURE STREQUAL "auto")
        AutodetectHostArchitecture()
        message(STATUS "Detected CPU: ${TARGET_ARCHITECTURE}")
    endif(TARGET_ARCHITECTURE STREQUAL "auto")

    add_feature(TARGET_ARCHITECTURE "Target CPU architecture")

    set(_ARCH_FLAGS )
    if(MSVC)
        message(WARNING "Windows support not implemented")
    elseif(CMAKE_CXX_COMPILER_IS_INTEL)
        set(OFA_map_knl "-xMIC-AVX512")
        set(OFA_map_cannonlake "-xCORE-AVX512")
        set(OFA_map_skylake-avx512 "-xCORE-AVX512")
        set(OFA_map_skylake "-xCORE-AVX2")
        set(OFA_map_broadwell "-xCORE-AVX2")
        set(OFA_map_haswell "-xCORE-AVX2")
        set(OFA_map_ivybridge "-xCORE-AVX-I")
        set(OFA_map_sandybridge "-xAVX")
        set(OFA_map_westmere "-xSSE4.2")
        set(OFA_map_nehalem "-xSSE4.2")
        set(OFA_map_penryn "-xSSSE3")
        set(OFA_map_merom "-xSSSE3")
        set(OFA_map_core2 "-xSSE3")
        set(OFA_map_kaby-lake "-xAVX")
        set(_ARCH_FLAGS "${OFA_map_${TARGET_ARCHITECTURE}}")
    else() # not MSVC and not ICC => GCC, Clang, Open64
        set(_ARCH_FLAGS "-march=native")
    endif()
    
    if(APPLE)
        add(_ARCH_FLAGS "-Wa,-W")
        add(_ARCH_FLAGS "-Wa,-q")
    endif(APPLE)

    set(${VAR} "${_ARCH_FLAGS}" PARENT_SCOPE)
endfunction()

# =============================================================================#
