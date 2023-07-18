#include "utils.h"

pair<vector<mdlp::labels_t>, map<string, int>> discretize(vector<mdlp::samples_t> &X, mdlp::labels_t &y, vector<string> features) {
    vector<mdlp::labels_t> Xd;
    map<string, int> maxes;
    auto fimdlp = mdlp::CPPFImdlp();
    for (int i = 0; i < X.size(); i++) {
        fimdlp.fit(X[i], y);
        mdlp::labels_t &xd = fimdlp.transform(X[i]);
        maxes[features[i]] = *max_element(xd.begin(), xd.end()) + 1;
        Xd.push_back(xd);
    }
    return {Xd, maxes};
}

tuple<vector<vector<int>>, vector<int>, vector<string>, string, map<string, vector<int>>>
loadFile(string name) {
    auto handler = ArffFiles();
    handler.load(PATH + static_cast<string>(name) + ".arff");
    // Get Dataset X, y
    vector<mdlp::samples_t> &X = handler.getX();
    mdlp::labels_t &y = handler.getY();
    // Get className & Features
    auto className = handler.getClassName();
    vector<string> features;
    for (auto feature: handler.getAttributes()) {
        features.push_back(feature.first);
    }
    // Discretize Dataset
    vector<mdlp::labels_t> Xd;
    map<string, int> maxes;
    tie(Xd, maxes) = discretize(X, y, features);
    maxes[className] = *max_element(y.begin(), y. end()) + 1;
    map<string, vector<int>> states;
    for (auto feature: features) {
        states[feature] = vector<int>(maxes[feature]);
    }
    states[className] = vector<int>(maxes[className]);
    return {Xd, y, features, className, states};
}