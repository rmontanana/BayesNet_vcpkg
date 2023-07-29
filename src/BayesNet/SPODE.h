#ifndef SPODE_H
#define SPODE_H
#include "Classifier.h"
namespace bayesnet {
    class SPODE : public Classifier {
    private:
        int root;
    protected:
        void train() override;
    public:
        SPODE(int root);
        virtual ~SPODE() {};
        vector<string> graph(string name = "SPODE") override;
    };
}
#endif