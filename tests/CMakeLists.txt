find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

# =============================================================================
# tmnn-owned unit tests
# =============================================================================
add_executable(tmnn_unit_tests
    main.cpp
    unit/tmnn/test_tmnn_core.cpp
    unit/tmnn/test_runtime_skeleton.cpp
    unit/tmnn/test_buffer_arena.cpp
    unit/tmnn/test_command_batch.cpp
    unit/tmnn/test_command_batch_error.mm
    unit/tmnn/test_metal_obj_ptr.mm
    unit/tmnn/test_metal_heap.mm
    unit/tmnn/test_owned_buffer_handle.cpp
    unit/tmnn/test_parameter_store.cpp
    unit/tmnn/test_pipeline_registry.cpp
    unit/tmnn/test_numerics_guard.cpp
    unit/tmnn/test_step_lane_coordinator.cpp
    unit/tmnn/test_extension_sdk.cpp
    unit/tmnn/test_factory_json.cpp
    unit/tmnn/test_autotune_manifest.cpp
    unit/tmnn/test_autotune_search.cpp
    unit/tmnn/test_field_evaluator.cpp
    unit/tmnn/test_family_policy.cpp
    unit/tmnn/test_network_planning.cpp
    unit/tmnn/test_schema_buffer_geometry.cpp
    unit/tmnn/test_standard_sdf_adapter.cpp
    unit/tmnn/test_multi_output_mlp_adapter.cpp
    unit/tmnn/test_dnl_adapter.cpp
    unit/tmnn/test_rmhe_adapter.cpp
    unit/tmnn/test_4d_adapter.cpp
    unit/tmnn/test_training_step_execution.cpp
    unit/tmnn/test_training_step_lifecycle.cpp
    unit/tmnn/test_kernel_generation.cpp
    unit/tmnn/test_default_trainer.cpp
)
target_link_libraries(tmnn_unit_tests PRIVATE
    tiny_metal_nn::tiny_metal_nn_runtime
    tiny_metal_nn::tiny_metal_nn_extensions
    tiny_metal_nn::tiny_metal_nn_kernels
    GTest::gtest
)
target_compile_features(tmnn_unit_tests PRIVATE cxx_std_23)

# Default per-test timeout. Sanitizer-instrumented binaries run 2–5× slower,
# so allow more wall-clock under TMNN_ENABLE_SANITIZERS to keep the slowest
# stress tests (e.g. StressRepeatedCreateDestroy) within budget.
if(TMNN_ENABLE_SANITIZERS)
    set(_TMNN_TEST_TIMEOUT 600)
else()
    set(_TMNN_TEST_TIMEOUT 120)
endif()

gtest_discover_tests(tmnn_unit_tests
    TEST_LIST TMNN_DISCOVERED_TESTS
    PROPERTIES TIMEOUT ${_TMNN_TEST_TIMEOUT}
)

# Apply labels at CTest runtime after gtest_discover_tests has materialized the
# individual discovered test names.
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/LabelSlowTests.cmake"
)

add_executable(tmnn_runtime_benchmarks
    benchmarks/tmnn_runtime_benchmarks.cpp
)
target_link_libraries(tmnn_runtime_benchmarks PRIVATE
    tiny_metal_nn::tiny_metal_nn_runtime
    tiny_metal_nn::tiny_metal_nn_extensions
)
target_compile_features(tmnn_runtime_benchmarks PRIVATE cxx_std_23)

# =============================================================================
# Package export smoke test
# Installs tiny_metal_nn to a temp prefix, then builds a consumer project
# that does find_package(tiny_metal_nn) and includes the public headers.
#
# When TMNN_ENABLE_SANITIZERS=ON, the sanitizer flags are forwarded to the
# consumer build as well: sanitizer-instrumented static libraries can only be
# linked against a sanitizer-instrumented consumer (otherwise the link step
# fails to find ASAN-wrapped runtime symbols). Forwarding the flags here
# tests the only valid deployment mode under sanitizers, instead of skipping
# the smoke test.
# =============================================================================
set(_TMNN_SMOKE_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/_tmnn_smoke_install")
set(_TMNN_SMOKE_BUILD_DIR   "${CMAKE_CURRENT_BINARY_DIR}/_tmnn_smoke_build")
set(_TMNN_SMOKE_CONFIGURE_ARGS
    -S ${CMAKE_CURRENT_SOURCE_DIR}/smoke
    -B ${_TMNN_SMOKE_BUILD_DIR}
    -DCMAKE_PREFIX_PATH=${_TMNN_SMOKE_INSTALL_DIR}
    -DCMAKE_CXX_STANDARD=23
)

if(DEFINED CMAKE_TOOLCHAIN_FILE)
    list(APPEND _TMNN_SMOKE_CONFIGURE_ARGS
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
    )
endif()

if(TMNN_ENABLE_SANITIZERS AND NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    list(APPEND _TMNN_SMOKE_CONFIGURE_ARGS
        "-DCMAKE_CXX_FLAGS_INIT=-fsanitize=address,undefined -fno-omit-frame-pointer"
        "-DCMAKE_EXE_LINKER_FLAGS_INIT=-fsanitize=address,undefined"
    )
endif()

add_test(
    NAME TinyMetalNNPackageSmokeTest_Install
    COMMAND ${CMAKE_COMMAND}
        -E env --unset=DESTDIR
        ${CMAKE_COMMAND}
            --install ${CMAKE_BINARY_DIR}
            --prefix ${_TMNN_SMOKE_INSTALL_DIR}
            --component Unspecified
)

add_test(
    NAME TinyMetalNNPackageSmokeTest_Configure
    COMMAND ${CMAKE_COMMAND} ${_TMNN_SMOKE_CONFIGURE_ARGS}
)
set_tests_properties(TinyMetalNNPackageSmokeTest_Configure PROPERTIES
    DEPENDS TinyMetalNNPackageSmokeTest_Install
    TIMEOUT 60
)

add_test(
    NAME TinyMetalNNPackageSmokeTest_Build
    COMMAND ${CMAKE_COMMAND} --build ${_TMNN_SMOKE_BUILD_DIR}
)
set_tests_properties(TinyMetalNNPackageSmokeTest_Build PROPERTIES
    DEPENDS TinyMetalNNPackageSmokeTest_Configure
    TIMEOUT 60
)

add_test(
    NAME TinyMetalNNPackageSmokeTest_Run
    COMMAND ${_TMNN_SMOKE_BUILD_DIR}/tmnn_smoke_test
)
set_tests_properties(TinyMetalNNPackageSmokeTest_Run PROPERTIES
    DEPENDS TinyMetalNNPackageSmokeTest_Build
    TIMEOUT 10
)
