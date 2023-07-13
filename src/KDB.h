#ifndef KDB_H
#define KDB_H
#include "BaseClassifier.h"
namespace bayesnet {
    using namespace std;
    using namespace torch;
    class KDB : public BaseClassifier {
    private:
        int k;
    protected:
        void train();
    public:
        KDB(int k);
    };
}
#endif