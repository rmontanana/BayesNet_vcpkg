#ifndef KDB_H
#define KDB_H
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
        void train() override;
    public:
        KDB(int k, float theta = 0.03);
        virtual ~KDB() {};
        vector<string> graph(string name = "KDB") override;
    };
}
#endif