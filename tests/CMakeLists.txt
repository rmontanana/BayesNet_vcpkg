if(ENABLE_TESTING)
    set(TEST_MAIN "unit_tests")
    include_directories(src)
    SET(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage --coverage")
    SET(GCC_COVERAGE_LINK_FLAGS    "--coverage")
    set(TEST_SOURCES main.cc ../sample/ArffFiles.cc ../sample/CPPFImdlp.cpp ../sample/Metrics.cpp 
    ../src/utils.cc ../src/Network.cc ../src/Node.cc ../src/Metrics.cc ../src/BaseClassifier.cc ../src/KDB.cc 
    ../src/TAN.cc ../src/SPODE.cc ../src/Ensemble.cc ../src/AODE.cc ../src/Mst.cc)
    add_executable(${TEST_MAIN} ${TEST_SOURCES})
    target_link_libraries(${TEST_MAIN} PUBLIC "${TORCH_LIBRARIES}"  Catch2::Catch2WithMain)
    add_test(NAME ${TEST_MAIN} COMMAND ${TEST_MAIN})
endif(ENABLE_TESTING)
