#ifndef TAN_H
#define TAN_H
#include "Classifier.h"
namespace bayesnet {
    using namespace std;
    using namespace torch;
    class TAN : public Classifier {
    private:
    protected:
        void train() override;
    public:
        TAN();
        vector<string> graph(string name = "TAN") override;
    };
}
#endif