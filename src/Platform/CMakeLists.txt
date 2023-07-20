include_directories(${BayesNet_SOURCE_DIR}/src/BayesNet)
include_directories(${BayesNet_SOURCE_DIR}/src/Platform)
add_executable(main Experiment.cc ArffFiles.cc CPPFImdlp.cpp Metrics.cpp platformUtils.cc)
target_link_libraries(main BayesNet "${TORCH_LIBRARIES}")