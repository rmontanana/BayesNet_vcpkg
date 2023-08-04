#ifndef PROPOSAL_H
#define PROPOSAL_H
#include <string>
#include <map>
#include <torch/torch.h>
#include "Network.h"
#include "CPPFImdlp.h"

namespace bayesnet {
    class Proposal {
    public:
        Proposal(vector<vector<int>>& Xv_, vector<int>& yv_, vector<string>& features_, string& className_);
        virtual ~Proposal();
    protected:
        void localDiscretizationProposal(map<string, vector<int>>& states, Network& model);
        void fit_local_discretization(map<string, vector<int>>& states, torch::Tensor& y);
        torch::Tensor Xf; // X continuous nxm tensor
        map<string, mdlp::CPPFImdlp*> discretizers;
    private:
        vector<string>& pFeatures;
        string& pClassName;
        vector<vector<int>>& Xv; // X discrete nxm vector
        vector<int>& yv;
    };
}

#endif