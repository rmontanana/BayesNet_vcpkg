#ifndef EXACTINFERENCE_H
#define EXACTINFERENCE_H
#include "Network.h"
#include "Node.h"
#include <map>
#include <vector>
#include <string>
using namespace std;

namespace bayesnet {
    class ExactInference {
    private:
        Network network;
        double computeFactor(map<string, int>&);
    public:
        ExactInference(Network&);
        vector<double> variableElimination(map<string, int>&);
    };
}
#endif