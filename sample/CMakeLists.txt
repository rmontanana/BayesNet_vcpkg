cmake_minimum_required(VERSION 3.20)

project(bayesnet_sample)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_library(BayesNet NAMES libBayesNet BayesNet libBayesNet.a  REQUIRED)
find_path(Bayesnet_INCLUDE_DIRS REQUIRED NAMES bayesnet)
find_library(FImdlp NAMES libfimdlp.a  PATHS REQUIRED)

message(STATUS "FImdlp=${FImdlp}")
message(STATUS "FImdlp_INCLUDE_DIRS=${FImdlp_INCLUDE_DIRS}")
message(STATUS "BayesNet=${BayesNet}")
message(STATUS "Bayesnet_INCLUDE_DIRS=${Bayesnet_INCLUDE_DIRS}")

include_directories(
    ../tests/lib/Files
    lib/json/include
    /usr/local/include
    /usr/local/include/fimdlp/
)

add_executable(bayesnet_sample sample.cc) 
target_link_libraries(bayesnet_sample ${FImdlp} "${TORCH_LIBRARIES}" "${BayesNet}")