#include "TAN.h"

namespace bayesnet {
    using namespace std;
    using namespace torch;

    TAN::TAN() : BaseClassifier(Network()) {}

    void TAN::train()
    {
        // 0. Add all nodes to the model
        addNodes();
        // 1. Compute mutual information between each feature and the class
        auto weights = metrics.conditionalEdge();
        // 2. Compute the maximum spanning tree
        auto mst = metrics.maximumSpanningTree(weights);
        // 3. Add edges from the maximum spanning tree to the model
        for (auto i = 0; i < mst.size(); ++i) {
            auto [from, to] = mst[i];
            model.addEdge(features[from], features[to]);
        }

    }

}