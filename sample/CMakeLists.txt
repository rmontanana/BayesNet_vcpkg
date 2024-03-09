cmake_minimum_required(VERSION 3.20)

project(bayesnet_sample)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_library(BayesNet NAMES BayesNet.a libBayesNet.a REQUIRED)

include_directories(
    lib/Files
    lib/mdlp
    lib/json/include
    /usr/local/include
)

add_subdirectory(lib/Files)
add_subdirectory(lib/mdlp)
add_executable(bayesnet_sample sample.cc) 
target_link_libraries(bayesnet_sample ArffFiles mdlp "${TORCH_LIBRARIES}" "${BayesNet}")