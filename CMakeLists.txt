cmake_minimum_required(VERSION 3.20)
project(lux-metal VERSION 0.1.0 LANGUAGES CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "lux-metal requires macOS/iOS")
endif()

# lux-metal is a PLUGIN (shared library) loaded by lux-accel at runtime
# It does NOT link against lux-gpu - only uses the backend API header

# Need lux-accel for the backend_api.h header only
# Try to find it, but allow manual path via LUXACCEL_INCLUDE_DIR
find_package(lux-accel CONFIG QUIET)
if(NOT lux-accel_FOUND)
  if(NOT LUXACCEL_INCLUDE_DIR)
    set(LUXACCEL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lux-accel/include;${CMAKE_CURRENT_SOURCE_DIR}/../install/include")
  endif()
endif()

add_library(lux_metal MODULE
  src/metal_plugin.mm
)

# No alias needed - this is a plugin, not a linkable library

target_compile_features(lux_metal PRIVATE cxx_std_17)

target_include_directories(lux_metal
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${CMAKE_CURRENT_SOURCE_DIR}/../lux-accel/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../install/include
)

# Only link Metal frameworks - no lux libraries at link time
target_link_libraries(lux_metal
  PRIVATE
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
)

set_source_files_properties(src/metal_plugin.mm PROPERTIES
  COMPILE_FLAGS "-fobjc-arc"
)

# Plugin settings
set_target_properties(lux_metal PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  # Plugin naming
  PREFIX ""
  SUFFIX ".plugin"
  # macOS bundle settings
  BUNDLE FALSE
)

include(GNUInstallDirs)

# Install plugin to lib directory
install(TARGETS lux_metal
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/lux/plugins
)

# End-to-end test: dlopen the plugin and run a real Metal vector add.
option(LUX_METAL_BUILD_TESTS "Build the lux-metal plugin smoke test" ON)
if(LUX_METAL_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_load.cpp")
  add_executable(lux_metal_test_load test/test_load.cpp)
  target_compile_features(lux_metal_test_load PRIVATE cxx_std_17)
  target_include_directories(lux_metal_test_load
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../lux-accel/include
      ${CMAKE_CURRENT_SOURCE_DIR}/../install/include
  )
  if(lux-accel_FOUND AND TARGET lux::luxaccel)
    target_include_directories(lux_metal_test_load PRIVATE
      $<TARGET_PROPERTY:lux::luxaccel,INTERFACE_INCLUDE_DIRECTORIES>)
  endif()
  target_link_libraries(lux_metal_test_load PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(lux_metal_test_load lux_metal)

  enable_testing()
  add_test(NAME lux_metal_load
           COMMAND lux_metal_test_load $<TARGET_FILE:lux_metal>)
endif()
