add_library(BayesNet utils.cc Network.cc Node.cc Metrics.cc BaseClassifier.cc KDB.cc TAN.cc SPODE.cc Ensemble.cc AODE.cc Mst.cc)
target_link_libraries(BayesNet "${TORCH_LIBRARIES}")