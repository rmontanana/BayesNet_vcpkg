#ifndef EXACTINFERENCE_H
#define EXACTINFERENCE_H
#include "Network.h"
#include "Factor.h"
#include "Node.h"
#include <map>
#include <vector>
#include <string>
using namespace std;

namespace bayesnet {
    class ExactInference {
    private:
        Network network;
        map<string, int> evidence;
        vector<Factor*> factors;
        vector<string> candidates; // variables to be removed
        void buildFactors();
        string nextCandidate(); // Return the next variable to eliminate using MinFill criterion
    public:
        ExactInference(Network&);
        ~ExactInference();
        void setEvidence(const map<string, int>&);
        vector<double> variableElimination();
    };
}
#endif