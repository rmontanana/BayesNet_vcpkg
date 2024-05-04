if(ENABLE_TESTING)
    include_directories(
        ${BayesNet_SOURCE_DIR}/lib/Files
        ${BayesNet_SOURCE_DIR}/lib/folding
        ${BayesNet_SOURCE_DIR}/lib/mdlp
        ${BayesNet_SOURCE_DIR}/lib/json/include
        ${BayesNet_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/configured_files/include
    )
    file(GLOB_RECURSE BayesNet_SOURCES "${BayesNet_SOURCE_DIR}/bayesnet/*.cc")
    add_executable(TestBayesNet TestBayesNetwork.cc TestBayesNode.cc TestBayesClassifier.cc 
        TestBayesModels.cc TestBayesMetrics.cc TestFeatureSelection.cc TestBoostAODE.cc
        TestUtils.cc TestBayesEnsemble.cc ${BayesNet_SOURCES})
    target_link_libraries(TestBayesNet PUBLIC "${TORCH_LIBRARIES}" ArffFiles mdlp PRIVATE Catch2::Catch2WithMain)
    add_test(NAME BayesNetworkTest COMMAND TestBayesNet)
    add_test(NAME Network COMMAND TestBayesNet "[Network]")
    add_test(NAME Node COMMAND TestBayesNet "[Node]")
    add_test(NAME Metrics COMMAND TestBayesNet "[Metrics]")
    add_test(NAME FeatureSelection COMMAND TestBayesNet "[FeatureSelection]")
    add_test(NAME Classifier COMMAND TestBayesNet "[Classifier]")
    add_test(NAME Ensemble COMMAND TestBayesNet "[Ensemble]")
    add_test(NAME Models COMMAND TestBayesNet "[Models]")
    add_test(NAME BoostAODE COMMAND TestBayesNet "[BoostAODE]")
    add_test(NAME A2DE COMMAND TestBayesNet "[A2DE]")
endif(ENABLE_TESTING)
