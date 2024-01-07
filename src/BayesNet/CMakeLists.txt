include_directories(
    ${BayesNet_SOURCE_DIR}/lib/mdlp
    ${BayesNet_SOURCE_DIR}/lib/Files
    ${BayesNet_SOURCE_DIR}/lib/folding
    ${BayesNet_SOURCE_DIR}/lib/json/include
    ${BayesNet_SOURCE_DIR}/src/BayesNet
    ${CMAKE_BINARY_DIR}/configured_files/include
)

add_library(BayesNet bayesnetUtils.cc Network.cc Node.cc BayesMetrics.cc Classifier.cc 
    KDB.cc TAN.cc SPODE.cc Ensemble.cc AODE.cc TANLd.cc KDBLd.cc SPODELd.cc AODELd.cc BoostAODE.cc 
    Mst.cc Proposal.cc CFS.cc FCBF.cc IWSS.cc FeatureSelect.cc )
target_link_libraries(BayesNet mdlp "${TORCH_LIBRARIES}")