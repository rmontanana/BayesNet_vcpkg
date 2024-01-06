include_directories(
    ${BayesNet_SOURCE_DIR}/lib/Files
    ${BayesNet_SOURCE_DIR}/lib/json/include
    ${BayesNet_SOURCE_DIR}/src/BayesNet
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)
add_library(PyWrap SHARED PyWrap.cc STree.cc ODTE.cc SVC.cc RandomForest.cc PyClassifier.cc)
#target_link_libraries(PyWrap ${Python3_LIBRARIES} "${TORCH_LIBRARIES}" ${LIBTORCH_PYTHON} Boost::boost Boost::python Boost::numpy xgboost::xgboost ArffFiles)
target_link_libraries(PyWrap ${Python3_LIBRARIES} "${TORCH_LIBRARIES}" ${LIBTORCH_PYTHON} Boost::boost Boost::python Boost::numpy ArffFiles)