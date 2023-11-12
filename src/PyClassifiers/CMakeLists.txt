include_directories(${PyWrap_SOURCE_DIR}/lib/Files)
include_directories(${PyWrap_SOURCE_DIR}/lib/json/include)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

add_library(PyWrap SHARED PyWrap.cc STree.cc ODTE.cc SVC.cc RandomForest.cc PyClassifier.cc)
#target_link_libraries(PyWrap ${Python3_LIBRARIES} "${TORCH_LIBRARIES}" ${LIBTORCH_PYTHON} Boost::boost Boost::python Boost::numpy xgboost::xgboost ArffFiles)
target_link_libraries(PyWrap ${Python3_LIBRARIES} "${TORCH_LIBRARIES}" ${LIBTORCH_PYTHON} Boost::boost Boost::python Boost::numpy ArffFiles)