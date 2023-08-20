#ifndef TAN_H
#define TAN_H
#include "Classifier.h"
namespace bayesnet {
    using namespace std;
    using namespace torch;
    class TAN : public Classifier {
    private:
    protected:
        void buildModel(const torch::Tensor& weights) override;
    public:
        TAN();
        virtual ~TAN() {};
        vector<string> graph(const string& name = "TAN") const override;
    };
}
#endif