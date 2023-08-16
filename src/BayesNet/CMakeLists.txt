include_directories(${BayesNet_SOURCE_DIR}/lib/mdlp)
include_directories(${BayesNet_SOURCE_DIR}/lib/Files)
include_directories(${BayesNet_SOURCE_DIR}/lib/featureselect)
include_directories(${BayesNet_SOURCE_DIR}/src/BayesNet)
include_directories(${BayesNet_SOURCE_DIR}/src/Platform)
add_library(BayesNet bayesnetUtils.cc Network.cc Node.cc BayesMetrics.cc Classifier.cc 
    KDB.cc TAN.cc SPODE.cc Ensemble.cc AODE.cc TANLd.cc KDBLd.cc SPODELd.cc AODELd.cc BoostAODE.cc 
    Mst.cc Proposal.cc ${BayesNet_SOURCE_DIR}/src/Platform/Models.cc)
target_link_libraries(BayesNet mdlp FeatureSelect "${TORCH_LIBRARIES}")