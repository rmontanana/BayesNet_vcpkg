#include "ExactInference.h"

namespace bayesnet {
    ExactInference::ExactInference(Network& net) : network(net), evidence(map<string, int>()), candidates(net.getFeatures()) {}
    void ExactInference::setEvidence(const map<string, int>& evidence)
    {
        this->evidence = evidence;
    }
    ExactInference::~ExactInference()
    {
        for (auto& factor : factors) {
            delete factor;
        }
    }
    void ExactInference::buildFactors()
    {
        for (auto node : network.getNodes()) {
            factors.push_back(node.second->toFactor());
        }
    }
    string ExactInference::nextCandidate()
    {
        string result = "";
        map<string, Node*> nodes = network.getNodes();
        int minFill = INT_MAX;
        for (auto candidate : candidates) {
            unsigned fill = nodes[candidate]->minFill();
            if (fill < minFill) {
                minFill = fill;
                result = candidate;
            }
        }
        return result;
    }
    vector<double> ExactInference::variableElimination()
    {
        vector<double> result;
        string candidate;
        buildFactors();
        // Eliminate evidence
        while ((candidate = nextCandidate()) != "") {
            // Erase candidate from candidates (Erase–remove idiom)
            candidates.erase(remove(candidates.begin(), candidates.end(), candidate), candidates.end());
            // sum-product variable elimination algorithm as explained in the book probabilistic graphical models
            // 1. Multiply all factors containing the variable
            vector<Factor*> factorsToMultiply;
            for (auto factor : factors) {
                if (factor->contains(candidate)) {
                    factorsToMultiply.push_back(factor);
                }
            }
            Factor* product = Factor::product(factorsToMultiply);
            // 2. Sum out the variable
            Factor* sum = product->sumOut(candidate);
            // 3. Remove factors containing the variable
            for (auto factor : factorsToMultiply) {
                factors.erase(remove(factors.begin(), factors.end(), factor), factors.end());
                delete factor;
            }
            // 4. Add the resulting factor to the list of factors
            factors.push_back(sum);

        }
        return result;
    }
}