include_directories(${BayesNet_SOURCE_DIR}/lib/mdlp)
add_library(BayesNet bayesnetUtils.cc Network.cc Node.cc BayesMetrics.cc Classifier.cc KDB.cc TAN.cc SPODE.cc Ensemble.cc AODE.cc TANNew.cc Mst.cc)
target_link_libraries(BayesNet mdlp "${TORCH_LIBRARIES}")