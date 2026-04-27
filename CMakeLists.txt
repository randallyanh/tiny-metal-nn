cmake_minimum_required(VERSION 3.20)

include(cmake/Vcpkg.cmake)

project(tiny-metal-nn LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)

if(APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 23)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
endif()

find_package(nlohmann_json CONFIG REQUIRED)

# =============================================================================
# Optional: AddressSanitizer + UndefinedBehaviorSanitizer
# =============================================================================
# Build with `-DTMNN_ENABLE_SANITIZERS=ON` to instrument all targets with ASAN
# and UBSAN. Off by default so ordinary builds and packaged releases stay
# unchanged. Sanitizers must be applied uniformly across compile and link
# units, hence the project-wide add_compile_options / add_link_options.
option(TMNN_ENABLE_SANITIZERS
    "Enable AddressSanitizer and UndefinedBehaviorSanitizer for all targets"
    OFF)

if(TMNN_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        message(WARNING "[tmnn] TMNN_ENABLE_SANITIZERS is not supported on MSVC; ignoring")
    else()
        message(STATUS "[tmnn] Sanitizers enabled (AddressSanitizer + UndefinedBehaviorSanitizer)")
        add_compile_options(
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
            -fno-sanitize-recover=all
        )
        add_link_options(
            -fsanitize=address,undefined
        )
    endif()
endif()

# =============================================================================
# Library: tiny_metal_nn_core (header-only public core)
# =============================================================================
add_library(tiny_metal_nn_core INTERFACE)
target_include_directories(tiny_metal_nn_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(tiny_metal_nn_core INTERFACE cxx_std_23)
target_link_libraries(tiny_metal_nn_core INTERFACE nlohmann_json::nlohmann_json)
add_library(tiny_metal_nn::tiny_metal_nn_core ALIAS tiny_metal_nn_core)

# =============================================================================
# Library: tiny_metal_nn_runtime (Metal runtime implementation)
# =============================================================================
set(TMNN_METAL_DEVICE_SOURCE src/tiny_metal_nn/runtime/metal_device_stub.cpp)
if(APPLE)
    set(TMNN_METAL_DEVICE_SOURCE src/tiny_metal_nn/runtime/metal_device.mm)
endif()

add_library(tiny_metal_nn_runtime STATIC
    src/tiny_metal_nn/runtime/autotune_manifest.cpp
    src/tiny_metal_nn/runtime/autotune_search.cpp
    src/tiny_metal_nn/runtime/metal_context.cpp
    ${TMNN_METAL_DEVICE_SOURCE}
    src/tiny_metal_nn/runtime/network_planner.cpp
    src/tiny_metal_nn/runtime/buffer_arena.cpp
    src/tiny_metal_nn/runtime/owned_buffer_handle.cpp
    src/tiny_metal_nn/runtime/command_batch.cpp
    src/tiny_metal_nn/runtime/metal_heap/metal_heap.mm
    src/tiny_metal_nn/runtime/parameter_store.cpp
    src/tiny_metal_nn/runtime/pipeline_registry.cpp
    src/tiny_metal_nn/runtime/numerics_guard.cpp
    src/tiny_metal_nn/runtime/step_lane_coordinator.cpp
    src/tiny_metal_nn/runtime/training_step_execution.cpp
    src/tiny_metal_nn/runtime/training_step_lifecycle.cpp
    src/tiny_metal_nn/runtime/default_trainer.cpp
)
target_link_libraries(tiny_metal_nn_runtime PUBLIC tiny_metal_nn::tiny_metal_nn_kernels)
target_link_libraries(tiny_metal_nn_runtime PUBLIC tiny_metal_nn::tiny_metal_nn_core)
target_include_directories(tiny_metal_nn_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(tiny_metal_nn_runtime PUBLIC cxx_std_23)

if(APPLE)
    target_link_libraries(tiny_metal_nn_runtime PUBLIC
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    set_source_files_properties(
        src/tiny_metal_nn/runtime/metal_device.mm
        src/tiny_metal_nn/runtime/metal_heap/metal_heap.mm PROPERTIES
        COMPILE_FLAGS "-fno-objc-arc"
        SKIP_PRECOMPILE_HEADERS ON
    )
endif()
add_library(tiny_metal_nn::tiny_metal_nn_runtime ALIAS tiny_metal_nn_runtime)

# =============================================================================
# Library: tiny_metal_nn_extensions (built-in training adapters)
# =============================================================================
add_library(tiny_metal_nn_extensions STATIC
    src/tiny_metal_nn/extension/dnl_adapter.cpp
    src/tiny_metal_nn/extension/four_d_adapter.cpp
    src/tiny_metal_nn/extension/multi_output_mlp_adapter.cpp
    src/tiny_metal_nn/extension/rmhe_adapter.cpp
    src/tiny_metal_nn/extension/standard_sdf_adapter.cpp
)
target_link_libraries(tiny_metal_nn_extensions PUBLIC tiny_metal_nn::tiny_metal_nn_core)
target_include_directories(tiny_metal_nn_extensions
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(tiny_metal_nn_extensions PUBLIC cxx_std_23)
add_library(tiny_metal_nn::tiny_metal_nn_extensions ALIAS tiny_metal_nn_extensions)

# =============================================================================
# Library: tiny_metal_nn_kernels (MSL kernel generation)
# =============================================================================
add_library(tiny_metal_nn_kernels STATIC
    src/tiny_metal_nn/kernels/kernel_spec.cpp
    src/tiny_metal_nn/kernels/mlp_kernel_emitter.cpp
    src/tiny_metal_nn/kernels/kernel_compiler.cpp
)
target_link_libraries(tiny_metal_nn_kernels PUBLIC tiny_metal_nn::tiny_metal_nn_core)
target_include_directories(tiny_metal_nn_kernels
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(tiny_metal_nn_kernels PUBLIC cxx_std_23)
add_library(tiny_metal_nn::tiny_metal_nn_kernels ALIAS tiny_metal_nn_kernels)

# =============================================================================
# Installation + CMake package export
# =============================================================================
install(
    TARGETS
        tiny_metal_nn_core
        tiny_metal_nn_runtime
        tiny_metal_nn_extensions
        tiny_metal_nn_kernels
    EXPORT tiny_metal_nnTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/tiny-metal-nn
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# metal_heap is a standalone, reusable module whose include path
# (tiny_metal_nn/runtime/metal_heap/metal_heap.h) is preserved on install
# so downstream consumers can adopt it without re-rooting the header.
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/tiny_metal_nn/runtime/metal_heap
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tiny_metal_nn/runtime
    FILES_MATCHING PATTERN "*.h"
)

set(TINY_METAL_NN_CMAKE_CONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/tiny_metal_nn")

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tiny_metal_nnConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/tiny_metal_nnConfig.cmake
    INSTALL_DESTINATION ${TINY_METAL_NN_CMAKE_CONFIG_DIR}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/tiny_metal_nnConfigVersion.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/tiny_metal_nnConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/tiny_metal_nnConfigVersion.cmake
    DESTINATION ${TINY_METAL_NN_CMAKE_CONFIG_DIR}
)

install(
    EXPORT tiny_metal_nnTargets
    FILE tiny_metal_nnTargets.cmake
    NAMESPACE tiny_metal_nn::
    DESTINATION ${TINY_METAL_NN_CMAKE_CONFIG_DIR}
)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    option(TMNN_BUILD_SAMPLES "Build tiny-metal-nn samples" ON)
else()
    option(TMNN_BUILD_SAMPLES "Build tiny-metal-nn samples" OFF)
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    option(TMNN_BUILD_TESTS "Build tiny-metal-nn smoke tests" ON)
else()
    option(TMNN_BUILD_TESTS "Build tiny-metal-nn smoke tests" OFF)
endif()

if(TMNN_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

if(TMNN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
