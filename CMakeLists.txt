cmake_minimum_required(VERSION 3.20)

project(BayesNet)
find_package(Torch REQUIRED)

if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(src)
add_subdirectory(sample)