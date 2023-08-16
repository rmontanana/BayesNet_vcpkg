#include "TAN.h"

namespace bayesnet {
    using namespace torch;

    TAN::TAN() : Classifier(Network()) {}

    void TAN::buildModel(const torch::Tensor& weights)
    {
        // 0. Add all nodes to the model
        addNodes();
        // 1. Compute mutual information between each feature and the class and set the root node
        // as the highest mutual information with the class
        auto mi = vector <pair<int, float >>();
        Tensor class_dataset = dataset.index({ -1, "..." });
        for (int i = 0; i < static_cast<int>(features.size()); ++i) {
            Tensor feature_dataset = dataset.index({ i, "..." });
            auto mi_value = metrics.mutualInformation(class_dataset, feature_dataset, weights);
            mi.push_back({ i, mi_value });
        }
        sort(mi.begin(), mi.end(), [](const auto& left, const auto& right) {return left.second < right.second;});
        auto root = mi[mi.size() - 1].first;
        // 2. Compute mutual information between each feature and the class
        auto weights_matrix = metrics.conditionalEdge(weights);
        cout << "*** Weights matrix ***\n";
        cout << weights_matrix << "\n";
        // 3. Compute the maximum spanning tree
        auto mst = metrics.maximumSpanningTree(features, weights_matrix, root);
        // 4. Add edges from the maximum spanning tree to the model
        for (auto i = 0; i < mst.size(); ++i) {
            auto [from, to] = mst[i];
            model.addEdge(features[from], features[to]);
        }
        // 5. Add edges from the class to all features
        for (auto feature : features) {
            model.addEdge(className, feature);
        }
    }
    vector<string> TAN::graph(const string& title) const
    {
        return model.graph(title);
    }
}