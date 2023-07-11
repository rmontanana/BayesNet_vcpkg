add_library(BayesNet Network.cc Node.cc Metrics.cc)
target_link_libraries(BayesNet "${TORCH_LIBRARIES}")