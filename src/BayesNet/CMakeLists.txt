include_directories(${BayesNet_SOURCE_DIR}/lib/mdlp)
include_directories(${BayesNet_SOURCE_DIR}/lib/Files)
add_library(BayesNet bayesnetUtils.cc Network.cc Node.cc BayesMetrics.cc Classifier.cc 
    KDB.cc TAN.cc SPODE.cc Ensemble.cc AODE.cc TANLd.cc KDBLd.cc SPODELd.cc AODELd.cc Mst.cc Proposal.cc)
target_link_libraries(BayesNet mdlp ArffFiles "${TORCH_LIBRARIES}")