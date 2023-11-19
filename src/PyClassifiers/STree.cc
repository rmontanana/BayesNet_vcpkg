#include "STree.h"

namespace pywrap {
    STree::STree() : PyClassifier("stree", "Stree")
    {
        validHyperparameters = { "C", "kernel", "max_iter", "max_depth", "random_state", "multiclass_strategy", "gamma", "max_features", "degree" };
    };
    std::string STree::graph()
    {
        return callMethodString("graph");
    }
} /* namespace pywrap */