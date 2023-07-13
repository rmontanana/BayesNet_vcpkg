#include "KDB.h"
#include "Metrics.hpp"

namespace bayesnet {
    using namespace std;
    using namespace torch;
    KDB::KDB(int k) : BaseClassifier(Network()), k(k) {}
    void KDB::train()
    {
        /*
        1. For each feature Xi, compute mutual information, I(X;C),
        where C is the class.
        2. Compute class conditional mutual information I(Xi;XjIC), f or each
        pair of features Xi and Xj, where i#j.
        3. Let the used variable list, S, be empty.
        4. Let the DAG network being constructed, BN, begin with a single
        class node, C.
        5. Repeat until S includes all domain features
        5.1. Select feature Xmax which is not in S and has the largest value
        I(Xmax;C).
        5.2. Add a node to BN representing Xmax.
        5.3. Add an arc from C to Xmax in BN.
        5.4. Add m = min(lSl,/c) arcs from m distinct features Xj in S with
        the highest value for I(Xmax;X,jC).
        5.5. Add Xmax to S.
        Compute the conditional probabilility infered by the structure of BN by
        using counts from DB, and output BN.
        */
        // 1. For each feature Xi, compute mutual information, I(X;C),
        // where C is the class.
        cout << "Computing mutual information between features and class" << endl;
        auto n_classes = states[className].size();
        auto metrics = Metrics(dataset, features, className, n_classes);
        for (auto i = 0; i < features.size(); i++) {
            Tensor firstFeature = X.index({ "...", i });
            Tensor secondFeature = y;
            double mi = metrics.mutualInformation(firstFeature, y);
            cout << "Mutual information between " << features[i] << " and " << className << " is " << mi << endl;

        }


    }
}