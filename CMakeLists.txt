# Top level CMakeLists.txt
#
# minimum required cmake version
#cmake_minimum_required( VERSION 3.21.0 FATAL_ERROR )
cmake_minimum_required( VERSION 3.17.0 FATAL_ERROR )

# Use latest policy
cmake_policy( VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} )

# project name
if( NX2_EXTENSION_360_VIDEO )
  project( NextSoftware2_360 VERSION 1.0 )
else()
  project( NextSoftware2 VERSION 1.0 )
endif()

# comment the following line, when creating a release
#set( NX2_VERSION_SUFFIX "dev" )
if( NX2_VERSION_SUFFIX )
  set( PROJECT_VERSION "${PROJECT_VERSION}-${NX2_VERSION_SUFFIX}" )
endif()

list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules" )
message( STATUS "CMAKE_MODULE_PATH: updating module path to: ${CMAKE_MODULE_PATH}" )

# we enable x86 intrinsics for all target architectures, because they are implemented through simd-everywhere on non-x86
set( NX2_ENABLE_X86_SIMD   TRUE  CACHE BOOL "enable x86 intrinsics" )
set( NX2_ENABLE_TRACING    FALSE CACHE BOOL "Compile in tracing functionality" )
set( NX2_ENABLE_CABAC_DUMP FALSE CACHE BOOL "Compile in bin dump functionality for CABAC retraining" )

include( NextSoftware2CompilerSupport )

# enable sse4.1 build for all source files for gcc and clang
if( NX2_ENABLE_X86_SIMD )
  if( UNIX OR MINGW )
    # when building for non-x86, but emulating simd using simd-everywhere (e.g. on ARM),
    # the x86-compiler flags are not understood by the compiler
    set_if_compiler_supports_flag( FLAG_msse41 -msse4.1 )
    add_compile_options( ${FLAG_msse41} )
  endif()

  if( NOT MSVC AND NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Emscripten" )
    check_missing_intrinsics()
  endif()

  #set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTARGET_SIMD_X86" )
  #set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTARGET_SIMD_X86" )

  # This potentially brings performance improvements for simd-everywhere, but I haven't
  # seen any, so it's disabled for now (GH)
  #set_if_compiler_supports_flag( FLAG_openmp_simd -fopenmp-simd )
  #add_compile_options( ${FLAG_openmp_simd} )

  message( STATUS "Enabling TARGET_SIMD_X86 even for non-x86 architectures" )
  add_compile_definitions( TARGET_SIMD_X86 )
endif()

# set common warning flags
add_compile_options( "$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wall;-Wno-unused-function;-Wno-enum-compare-switch;-Wno-unknown-attributes;-Wno-unused-but-set-variable;-Wno-unused-const-variable>" )
add_compile_options( "$<$<CXX_COMPILER_ID:GNU>:-Wall;-Wno-unused-function;-Wno-sign-compare;-fdiagnostics-show-option;-Wno-ignored-attributes>" )
add_compile_options( "$<$<CXX_COMPILER_ID:MSVC>:/W4;/wd4100;/wd4127;/wd4244;/wd4245;/wd4310;/wd4389;/wd4456;/wd4457;/wd4458;/wd4459;/wd4505;/wd4701;/wd4702;/wd4703;/wd4706;/wd4996>" )

# Link-time optimization. May improve runtime by a few percent but is slower to build.
# If you need a rapid build time and nearly-as-fast runtime then set this OFF or use the RelWithDebInfo mode.
set( NX2_ENABLE_LINK_TIME_OPT ON CACHE BOOL "Enable link-time optimizations in Release and MinSizeRel modes" )
include( CheckIPOSupported )
check_ipo_supported( RESULT supported OUTPUT error )
if ( NX2_ENABLE_LINK_TIME_OPT AND supported )
  message( STATUS "Enabling link-time optimizations" )
  set( CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON )
  set( CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON )
endif()

set( NX2_ENABLE_WARNINGS_AS_WERROR        ON  CACHE BOOL   "Treat warnings as errors (-Werror or /WX)" )

set( NX2_USE_ADDRESS_SANITIZER            OFF CACHE BOOL   "Enable or disable address sanitizer" )

if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
  # enable address sanitizer
  # enable thread sanitizer
  set( NX2_USE_THREAD_SANITIZER           OFF CACHE BOOL   "Enable or disable thread sanitizer" )
  # add -march=native
  set( NX2_OPT_TARGET_ARCH                ""  CACHE STRING "Enable or disable building with architecture specific optimization" )
endif()

if( NX2_ENABLE_WARNINGS_AS_WERROR AND (NX2_USE_ADDRESS_SANITIZER OR NX2_USE_THREAD_SANITIZER) )
  # Don't combine -Werror with santitizers
  # From the GCC docs (https://gcc.gnu.org/onlinedocs/gcc-13.2.0/gcc/Instrumentation-Options.html):
  #   Note that sanitizers tend to increase the rate of false positive warnings, most notably those around -Wmaybe-uninitialized.
  #   We recommend against combining -Werror and [the use of] sanitizers.
  message( NOTICE "Disabling -Werror for santizer builds, because it leads to false positive warnings." )
  set( NX2_ENABLE_WARNINGS_AS_WERROR OFF )
endif()

set(PATH_TO_DIRECTORY_ROOT_SADL "../../../thirdparty/sadl")

set( NX2_EXTENSION_360_VIDEO OFF CACHE BOOL "If NX2_EXTENSION_360_VIDEO is on, 360Lib will be added" )
set( NX2_EXTENSION_HDRTOOLS OFF CACHE BOOL "If NX2_EXTENSION_HDRTOOLS is on, HDRLib will be added" )

if (DEFINED NX2_ENABLE_TRACING)
  set( NX2_ENABLE_TRACING OFF CACHE BOOL "ENABLE_TRACING will be set to this value" )
endif()

if (DEFINED NX2_ENABLE_CABAC_DUMP)
  set( NX2_ENABLE_CABAC_DUMP OFF CACHE BOOL "ENABLE_CABAC_DUMP will be set to this value" )
endif()

set( NX2_TOPLEVEL_OUTPUT_DIRS              ON  CACHE BOOL   "Put build artifacts into ${CMAKE_SOURCE_DIR}/{bin,lib}/" )
if( NX2_TOPLEVEL_OUTPUT_DIRS )
  # Using CMake's default library name convention which is the same for all configurations.
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG          "${CMAKE_SOURCE_DIR}/lib/debug" )
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE        "${CMAKE_SOURCE_DIR}/lib/release" )
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/lib/relwithdebinfo" )
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL     "${CMAKE_SOURCE_DIR}/lib/minsizerel" )

  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG          "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG}" )
  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE        "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE}" )
  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO}" )
  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL     "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL}" )

  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${CMAKE_SOURCE_DIR}/bin/debug" )
  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${CMAKE_SOURCE_DIR}/bin/release" )
  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/bin/relwithdebinfo" )
  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${CMAKE_SOURCE_DIR}/bin/minsizerel" )
else()
  # Put build artifacts below build-directory. (Don't need to add {debug,release,relwithdebinfo,minsizerel}-{static,shared}/)
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )
  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )
  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" )
endif()

# set default CMAKE_BUILD_TYPE to Release if not set
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE )
endif()

# set c++17
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

# compile everything position independent (even static libraries)
set( CMAKE_POSITION_INDEPENDENT_CODE TRUE )

# use folders in IDEs for projects (e.g. lib sample app test)
set_property( GLOBAL PROPERTY USE_FOLDERS ON )

# Enable multithreading
find_package( Threads REQUIRED )

# set _WIN32_WINNT
if( WIN32 )
  # set _WIN32_WINT version global
  add_definitions( -D_WIN32_WINNT=0x0600 )
endif()

# enable parallel build for Visual Studio
if( MSVC )
  # add compile options
  add_compile_options( "/MP" )
  add_compile_options( "/EHsc" )
endif()

# set address sanitizer compiler arguments
if( NX2_USE_ADDRESS_SANITIZER )
  if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
    message( STATUS "enabled GNU/Clang address sanitizer" )
    # add compile options
    add_compile_options( "-fsanitize=address" )
    add_compile_options( "-fsanitize=undefined" )
    add_compile_options( "-fsanitize=leak" )
    add_compile_options( "-fno-omit-frame-pointer" )
    add_link_options( "-fsanitize=address" )
    add_link_options( "-fsanitize=undefined" )
    add_link_options( "-fsanitize=leak" )
    add_link_options( "-fno-omit-frame-pointer" )
  elseif(MSVC)  
    message( STATUS "enabled MSVC address sanitizer" )
    add_compile_options( "/fsanitize=address" )
  endif()  
endif()

# set thread sanitizer compiler arguments
if( NX2_USE_THREAD_SANITIZER )
  # add compile options
  add_compile_options( "-fsanitize=thread" )
  add_link_options( "-fsanitize=thread" )
endif()

if( NX2_ENABLE_WARNINGS_AS_WERROR )
    add_compile_options( "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Werror>" )
    add_compile_options( "$<$<CXX_COMPILER_ID:MSVC>:/WX>" )
endif()

if( NX2_ENABLE_X86_SIMD )
  if( ( UNIX OR MINGW ) AND NX2_OPT_TARGET_ARCH )
    set( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -march=${NX2_OPT_TARGET_ARCH} -mtune=${NX2_OPT_TARGET_ARCH}" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${NX2_OPT_TARGET_ARCH} -mtune=${NX2_OPT_TARGET_ARCH}" )
  endif()
endif()

# use ccache
find_program( CCACHE_FOUND ccache )
if( CCACHE_FOUND )
  message( STATUS "ccache found. using it." )
  set_property( GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache )
  set_property( GLOBAL PROPERTY RULE_LAUNCH_LINK ccache )
endif()

# add needed subdirectories
add_subdirectory( "source/Lib/CommonLib" )
if( NX2_EXTENSION_360_VIDEO )
  add_subdirectory( "source/Lib/Lib360" )
  add_subdirectory( "source/Lib/AppEncHelper360" )
endif()
if ( NX2_EXTENSION_HDRTOOLS )
  add_subdirectory( "source/Lib/HDRLib")
endif()
add_subdirectory( "source/Lib/DecoderLib" )
add_subdirectory( "source/Lib/EncoderLib" )
add_subdirectory( "source/Lib/Utilities" )

add_subdirectory( "source/App/DecoderApp" )
add_subdirectory( "source/App/EncoderApp" )
add_subdirectory( "source/App/SEIRemovalApp" )
add_subdirectory( "source/App/Parcat" )
add_subdirectory( "source/App/CabacTraining" )
if( NX2_EXTENSION_360_VIDEO )
  add_subdirectory( "source/App/utils/360ConvertApp" )
endif()
