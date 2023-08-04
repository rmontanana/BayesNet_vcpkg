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
        Proposal(vector<vector<int>>& Xv_, vector<int>& yv_);
        virtual ~Proposal() = default;
    protected:
        void localDiscretizationProposal(Network& model, vector<string>& features, string className, map<string, vector<int>>& states);
        void fit_local_discretization(vector<string>& features, string className, map<string, vector<int>>& states, torch::Tensor& y);
        torch::Tensor Xf; // X continuous nxm tensor
        map<string, mdlp::CPPFImdlp*> discretizers;
    private:
        vector<vector<int>>& Xv; // X discrete nxm vector
        vector<int>& yv;
    };
}

#endif