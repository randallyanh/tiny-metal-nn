add_executable(mlp_learning_an_image
    mlp_learning_an_image.cpp
)
target_link_libraries(mlp_learning_an_image PRIVATE
    tiny_metal_nn::tiny_metal_nn_runtime
    tiny_metal_nn::tiny_metal_nn_extensions
)
target_compile_features(mlp_learning_an_image PRIVATE cxx_std_23)
