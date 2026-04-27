cmake_minimum_required(VERSION 3.20)
project(tmnn_smoke LANGUAGES CXX)

find_package(tiny_metal_nn CONFIG REQUIRED)

add_executable(tmnn_smoke_test smoke_test.cpp)
target_link_libraries(tmnn_smoke_test PRIVATE
    tiny_metal_nn::tiny_metal_nn_core
    tiny_metal_nn::tiny_metal_nn_runtime
)
target_compile_features(tmnn_smoke_test PRIVATE cxx_std_23)
