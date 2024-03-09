if(ENABLE_TESTING)
    set(TEST_BAYESNET "unit_tests_bayesnet")
    include_directories(
        ${BayesNet_SOURCE_DIR}/lib/Files
        ${BayesNet_SOURCE_DIR}/lib/mdlp
        ${BayesNet_SOURCE_DIR}/lib/folding
        ${BayesNet_SOURCE_DIR}/lib/json/include
        ${BayesNet_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/configured_files/include/bayesnet
    )
    file(GLOB_RECURSE BayesNet_SOURCES "${BayesNet_SOURCE_DIR}/bayesnet/*.cc")
    set(TEST_SOURCES_BAYESNET TestBayesModels.cc TestBayesNetwork.cc TestBayesMetrics.cc TestUtils.cc ${BayesNet_SOURCES})
    add_executable(${TEST_BAYESNET} ${TEST_SOURCES_BAYESNET})
    target_link_libraries(${TEST_BAYESNET} PUBLIC "${TORCH_LIBRARIES}" ArffFiles mdlp Catch2::Catch2WithMain )
    add_test(NAME ${TEST_BAYESNET} COMMAND ${TEST_BAYESNET})
endif(ENABLE_TESTING)
