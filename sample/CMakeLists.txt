include_directories(${BayesNet_SOURCE_DIR}/src)
add_executable(main main.cc ArffFiles.cc CPPFImdlp.cpp Metrics.cpp)
target_link_libraries(main BayesNet "${TORCH_LIBRARIES}")