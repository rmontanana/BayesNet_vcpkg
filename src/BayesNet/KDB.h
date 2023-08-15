#ifndef KDB_H
#define KDB_H
#include <torch/torch.h>
#include "Classifier.h"
#include "bayesnetUtils.h"
namespace bayesnet {
    using namespace std;
    using namespace torch;
    class KDB : public Classifier {
    private:
        int k;
        float theta;
        void add_m_edges(int idx, vector<int>& S, Tensor& weights);
    protected:
        void buildModel(const torch::Tensor& weights) override;
    public:
        explicit KDB(int k, float theta = 0.03);
        virtual ~KDB() {};
        vector<string> graph(const string& name = "KDB") const override;
    };
}
#endif