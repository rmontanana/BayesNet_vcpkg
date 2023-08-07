#ifndef PROPOSAL_H
#define PROPOSAL_H
#include <string>
#include <map>
#include <torch/torch.h>
#include "Network.h"
#include "CPPFImdlp.h"
#include "Classifier.h"

namespace bayesnet {
    class Proposal {
    public:
        Proposal(torch::Tensor& pDataset, vector<string>& features_, string& className_);
        virtual ~Proposal();
    protected:
        torch::Tensor prepareX(torch::Tensor& X);
        void localDiscretizationProposal(map<string, vector<int>>& states, Network& model);
        void fit_local_discretization(map<string, vector<int>>& states, torch::Tensor& y);
        torch::Tensor Xf; // X continuous nxm tensor
        torch::Tensor y; // y discrete nx1 tensor
        map<string, mdlp::CPPFImdlp*> discretizers;
    private:
        torch::Tensor& pDataset; // (n+1)xm tensor
        vector<string>& pFeatures;
        string& pClassName;
    };
}

#endif  