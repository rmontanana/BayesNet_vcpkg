include_directories(${BayesNet_SOURCE_DIR}/src/Platform)
include_directories(${BayesNet_SOURCE_DIR}/src/BayesNet)
add_executable(sample sample.cc ${BayesNet_SOURCE_DIR}/src/Platform/ArffFiles.cc ${BayesNet_SOURCE_DIR}/src/Platform/CPPFImdlp.cpp ${BayesNet_SOURCE_DIR}/src/Platform/Metrics.cpp ${BayesNet_SOURCE_DIR}/src/Platform/typesFImdlp.h ${BayesNet_HEADERS})
target_link_libraries(sample BayesNet "${TORCH_LIBRARIES}")