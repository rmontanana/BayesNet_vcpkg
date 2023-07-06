add_library(BayesNet Network.cc Node.cc)
target_link_libraries(BayesNet "${TORCH_LIBRARIES}")