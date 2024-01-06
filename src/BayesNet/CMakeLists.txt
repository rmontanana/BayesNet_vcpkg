include_directories(
    ${BayesNet_SOURCE_DIR}/lib/mdlp
    ${BayesNet_SOURCE_DIR}/lib/Files
    ${BayesNet_SOURCE_DIR}/lib/json/include
    ${BayesNet_SOURCE_DIR}/src/BayesNet
    ${BayesNet_SOURCE_DIR}/src/Platform
    ${BayesNet_SOURCE_DIR}/src/PyClassifiers
    ${Python3_INCLUDE_DIRS}
)

add_library(BayesNet bayesnetUtils.cc Network.cc Node.cc BayesMetrics.cc Classifier.cc 
    KDB.cc TAN.cc SPODE.cc Ensemble.cc AODE.cc TANLd.cc KDBLd.cc SPODELd.cc AODELd.cc BoostAODE.cc 
    Mst.cc Proposal.cc CFS.cc FCBF.cc IWSS.cc FeatureSelect.cc ${BayesNet_SOURCE_DIR}/src/Platform/Models.cc)
target_link_libraries(BayesNet mdlp "${TORCH_LIBRARIES}")