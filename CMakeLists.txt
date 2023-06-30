cmake_minimum_required(VERSION 3.20)

project(bayesnet)
find_package(Torch REQUIRED)

if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_library(BayesNet Node.cc Network.cc)
add_executable(BayesNet main.cc ArffFiles.cc Node.cc Network.cc CPPFImdlp.cpp Metrics.cpp)
target_link_libraries(BayesNet "${TORCH_LIBRARIES}")