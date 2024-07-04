include_directories(
    ${BayesNet_SOURCE_DIR}/lib/mdlp/src
    ${BayesNet_SOURCE_DIR}/lib/folding
    ${BayesNet_SOURCE_DIR}/lib/json/include
    ${BayesNet_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/configured_files/include
)

file(GLOB_RECURSE Sources "*.cc")

add_library(BayesNet ${Sources})
target_link_libraries(BayesNet mdlp "${TORCH_LIBRARIES}")
