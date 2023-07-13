add_library(BayesNet Network.cc Node.cc Metrics.cc BaseClassifier.cc KDB.cc)
target_link_libraries(BayesNet "${TORCH_LIBRARIES}")