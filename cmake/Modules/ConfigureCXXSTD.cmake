
# - Include guard
if(__configurecxxstd_isloaded)
  return()
endif()
set(__configurecxxstd_isloaded YES)

include(MacroUtilities)

# make C++11 required for compilers that recognise standards
set(CMAKE_CXX_STANDARD "11")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# - Rough n'Ready setup of CXX compile features for Intel
#
#-----------------------------------------------------------------------
# Add compile features for Intel - should eventually be placed
# into a module, as it will need exporting for use by clients
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  # Now version selection, focus on version 15 and above as that is the
  # baseline - means that this isn't comprehensive for lower versions

  # c++98 should be supported for all(?) versions we may encounter, and
  # make it the default as required for compilers that recognise standards
  set(CMAKE_CXX_STANDARD_DEFAULT "11")

  if(NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15.0))

    set(CMAKE_CXX11_STANDARD_COMPILE_OPTION "-std=c++11")
    set(CMAKE_CXX11_EXTENSION_COMPILE_OPTION "-std=gnu++11")

    # 15 and higher also allow c++14...
    set(CMAKE_CXX14_STANDARD_COMPILE_OPTION "-std=c++14")
    set(CMAKE_CXX14_EXTENSION_COMPILE_OPTION "-std=gnu++14")
  endif()

  # Now the compile features... Eventually want to do these in the same
  # way as CMake does (using compile time checks on compiler version
  # etc - may also allow checks on GCC version in use), but for now
  # *assume* that the underlying library will support.
  if(NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16.0))
    set(CMAKE_CXX14_COMPILE_FEATURES
      cxx_contextual_conversions
      cxx_generic_lambdas
      cxx_aggregate_default_initializers
      cxx_attribute_deprecated
      cxx_digit_separators
      )
  endif()

  if(NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15.0))
    set(CMAKE_CXX14_COMPILE_FEATURES
      ${CMAKE_CXX14_COMPILE_FEATURES}
      cxx_binary_literals
      cxx_decltype_auto #cxx_return_type_deduction?
      cxx_lambda_init_captures
      )
    if(NOT WIN32)
      list(APPEND CMAKE_CXX14_COMPILE_FEATURES cxx_attribute_deprecated)
    endif()

    set(CMAKE_CXX11_COMPILE_FEATURES
      cxx_alias_templates
      cxx_alignas
      cxx_alignof
      cxx_attributes
      cxx_auto_type
      cxx_constexpr
      cxx_decltype_incomplete_return_types
      cxx_decltype
      cxx_default_function_template_args
      cxx_defaulted_functions
      cxx_defaulted_move_initializers
      cxx_delegating_constructors
      cxx_deleted_functions
      cxx_enum_forward_declarations
      cxx_explicit_conversions
      cxx_extended_friend_declarations
      cxx_extern_templates
      cxx_final
      cxx_func_identifier
      cxx_generalized_initializers
      cxx_inheriting_constructors
      cxx_inline_namespaces
      cxx_lambdas
      cxx_local_type_template_args
      cxx_long_long_type
      cxx_noexcept
      cxx_nonstatic_member_init
      cxx_nullptr
      cxx_override
      cxx_range_for
      cxx_raw_string_literals
      cxx_reference_qualified_functions
      cxx_right_angle_brackets
      cxx_rvalue_references
      cxx_sizeof_member
      cxx_static_assert
      cxx_strong_enums
      cxx_trailing_return_types
      cxx_unicode_literals
      cxx_uniform_initialization # Not explicit in Intel notes but should do...
      cxx_user_literals
      cxx_variadic_macros
      cxx_variadic_templates
      )

    if(UNIX)
      list(APPEND CMAKE_CXX11_COMPILE_FEATURES
        cxx_unrestricted_unions # Linux/OSX only
        cxx_thread_local # Linux/OSX only
        )
    endif()
  endif()

  set(CMAKE_CXX_COMPILE_FEATURES
    ${CMAKE_CXX11_COMPILE_FEATURES}
    ${CMAKE_CXX14_COMPILE_FEATURES}
    )
endif()


#-----------------------------------------------------------------------
# Configure/Select C++ Standard
# Require at least C++11 with no extensions and the following features
set(CMAKE_CXX_EXTENSIONS ON)

set(${PROJECT_NAME}_TARGET_COMPILE_FEATURES
  cxx_alias_templates
  cxx_auto_type
  cxx_delegating_constructors
  cxx_enum_forward_declarations
  cxx_explicit_conversions
  cxx_final
  cxx_lambdas
  cxx_nullptr
  cxx_override
  cxx_range_for
  cxx_strong_enums
  cxx_uniform_initialization
  # Features that MSVC 18.0 cannot support but in list of coding
  # guidelines - to be required once support for that compiler is dropped.
  # Version 10.2 is coded without these being required.
  #cxx_deleted_functions
  #cxx_generalized_initializers
  cxx_constexpr
  #cxx_inheriting_constructors
)

# - BUILD_CXXSTD
# Choose C++ Standard to build against from supported list. Allow user
# to supply it as a simple year or as 'c++XY'. If the latter, post process
# to remove the 'c++'
# Mark as advanced because most users will not need it
enum_option(BUILD_CXXSTD
  DOC "C++ Standard to compile against"
  VALUES 11 14 17 c++11 c++14 c++17
  CASE_INSENSITIVE
)

string(REGEX REPLACE "^c\\+\\+" "" BUILD_CXXSTD "${BUILD_CXXSTD}")
mark_as_advanced(BUILD_CXXSTD)
add_feature(BUILD_CXXSTD "Compiling against C++ Standard '${BUILD_CXXSTD}'")

# If a standard higher than 11 has been selected, check that compiler has
# at least one feature from that standard and append these to the required
# feature list
if(CMAKE_CXX${BUILD_CXXSTD}_COMPILE_FEATURES)
    list(APPEND ${PROJECT_NAME}_TARGET_COMPILE_FEATURES ${CMAKE_CXX${BUILD_CXXSTD}_COMPILE_FEATURES})
else()
    message(FATAL_ERROR "${PROJECT_NAME} requested to be compiled against C++ standard '${BUILD_CXXSTD}'\nbut detected compiler '${CMAKE_CXX_COMPILER_ID}', version '${CMAKE_CXX_COMPILER_VERSION}'\ndoes not support any features of that standard")
endif()

set(BUILD_CXXSTD "c++${BUILD_CXXSTD}")

if(CMAKE_C_COMPILER_IS_GNU)
    add(CMAKE_C_FLAGS_EXTRA "-std=c99")
endif(CMAKE_C_COMPILER_IS_GNU)
