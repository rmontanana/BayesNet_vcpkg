include_directories(
    ${BayesNet_SOURCE_DIR}/src
    ${BayesNet_SOURCE_DIR}/src/classifiers
    ${BayesNet_SOURCE_DIR}/src/ensembles
    ${BayesNet_SOURCE_DIR}/src/bayesian_network
    ${BayesNet_SOURCE_DIR}/src/utils
    ${BayesNet_SOURCE_DIR}/src/feature_selection
    ${BayesNet_SOURCE_DIR}/lib/Files
    ${BayesNet_SOURCE_DIR}/lib/mdlp
    ${BayesNet_SOURCE_DIR}/lib/json/include
    ${CMAKE_BINARY_DIR}/configured_files/include
)
add_executable(bayesnet_sample sample.cc) 
target_link_libraries(bayesnet_sample ArffFiles BayesNet)