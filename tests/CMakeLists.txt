if(ENABLE_TESTING)
    set(TEST_MAIN "unit_tests")
    set(TEST_SOURCES BayesModels.cc BayesNetwork.cc ../sample/ArffFiles.cc ../sample/CPPFImdlp.cpp ../sample/Metrics.cpp 
    ../src/utils.cc ../src/Network.cc ../src/Node.cc ../src/Metrics.cc ../src/BaseClassifier.cc ../src/KDB.cc 
    ../src/TAN.cc ../src/SPODE.cc ../src/Ensemble.cc ../src/AODE.cc ../src/Mst.cc utils.cc utils.h)
    add_executable(${TEST_MAIN} ${TEST_SOURCES})
    target_link_libraries(${TEST_MAIN} PUBLIC "${TORCH_LIBRARIES}"  Catch2::Catch2WithMain)
    add_test(NAME ${TEST_MAIN} COMMAND ${TEST_MAIN})
endif(ENABLE_TESTING)
