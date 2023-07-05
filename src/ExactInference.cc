#include "ExactInference.h"

namespace bayesnet {
    ExactInference::ExactInference(Network& net) : network(net) {}
    double ExactInference::computeFactor(map<string, int>& completeEvidence)
    {
        double result = 1.0;
        for (auto node : network.getNodes()) {
            result *= node.second->getFactorValue(completeEvidence);
        }
        return result;
    }
    vector<double> ExactInference::variableElimination(map<string, int>& evidence)
    {
        vector<double> result;
        string candidate;
        int classNumStates = network.getClassNumStates();
        for (int i = 0; i < classNumStates; ++i) {
            result.push_back(1.0);
            auto complete_evidence = map<string, int>(evidence);
            complete_evidence[network.getClassName()] = i;
            result[i] = computeFactor(complete_evidence);
        }
        // Normalize result
        auto sum = accumulate(result.begin(), result.end(), 0.0);
        for (int i = 0; i < result.size(); ++i) {
            result[i] /= sum;
        }
        return result;
    }
}