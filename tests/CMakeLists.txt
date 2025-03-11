if(ENABLE_TESTING)
    include_directories(
        ${BayesNet_SOURCE_DIR}/tests/lib/Files
        ${BayesNet_SOURCE_DIR}/lib/folding
        ${BayesNet_SOURCE_DIR}/lib/mdlp/src
        ${BayesNet_SOURCE_DIR}/lib/log
        ${BayesNet_SOURCE_DIR}/lib/json/include
        ${BayesNet_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/configured_files/include
    )
    file(GLOB_RECURSE BayesNet_SOURCES "${BayesNet_SOURCE_DIR}/bayesnet/*.cc")
    add_executable(TestBayesNet TestBayesNetwork.cc TestBayesNode.cc TestBayesClassifier.cc 
        TestBayesModels.cc TestBayesMetrics.cc TestFeatureSelection.cc TestBoostAODE.cc TestXBAODE.cc TestA2DE.cc TestWA2DE.cc
        TestUtils.cc TestBayesEnsemble.cc TestModulesVersions.cc TestBoostA2DE.cc TestMST.cc ${BayesNet_SOURCES})
    target_link_libraries(TestBayesNet PUBLIC "${TORCH_LIBRARIES}" fimdlp PRIVATE Catch2::Catch2WithMain)
    add_test(NAME BayesNetworkTest COMMAND TestBayesNet)
    add_test(NAME A2DE COMMAND TestBayesNet "[A2DE]")
    add_test(NAME WA2DE COMMAND TestBayesNet "[WA2DE]")
    add_test(NAME BoostA2DE COMMAND TestBayesNet "[BoostA2DE]")
    add_test(NAME BoostAODE COMMAND TestBayesNet "[BoostAODE]")
    add_test(NAME XBAODE COMMAND TestBayesNet "[XBAODE]")
    add_test(NAME Classifier COMMAND TestBayesNet "[Classifier]")
    add_test(NAME Ensemble COMMAND TestBayesNet "[Ensemble]")
    add_test(NAME FeatureSelection COMMAND TestBayesNet "[FeatureSelection]")
    add_test(NAME Metrics COMMAND TestBayesNet "[Metrics]")
    add_test(NAME Models COMMAND TestBayesNet "[Models]")
    add_test(NAME Modules COMMAND TestBayesNet "[Modules]")
    add_test(NAME Network COMMAND TestBayesNet "[Network]")
    add_test(NAME Node COMMAND TestBayesNet "[Node]")
    add_test(NAME MST COMMAND TestBayesNet "[MST]")
endif(ENABLE_TESTING)
