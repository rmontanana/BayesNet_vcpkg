#include "TAN.h"

namespace bayesnet {
    using namespace std;
    using namespace torch;

    TAN::TAN() : BaseClassifier(Network()) {}

    void TAN::train()
    {
        // 0. Add all nodes to the model
        addNodes();
        // 1. Compute mutual information between each feature and the class and set the root node
        // as the highest mutual information with the class
        auto mi = vector <pair<int, float >>();
        Tensor class_dataset = dataset.index({ "...", -1 });
        for (int i = 0; i < static_cast<int>(features.size()); ++i) {
            Tensor feature_dataset = dataset.index({ "...", i });
            auto mi_value = metrics.mutualInformation(class_dataset, feature_dataset);
            mi.push_back({ i, mi_value });
        }
        sort(mi.begin(), mi.end());
        auto root = mi[mi.size() - 1].first;
        // 2. Compute mutual information between each feature and the class
        auto weights = metrics.conditionalEdge();
        // 3. Compute the maximum spanning tree
        auto mst = metrics.maximumSpanningTree(root, weights);
        // 4. Add edges from the maximum spanning tree to the model
        for (auto i = 0; i < mst.size(); ++i) {
            auto [from, to] = mst[i];
            model.addEdge(features[from], features[to]);
        }
    }
}