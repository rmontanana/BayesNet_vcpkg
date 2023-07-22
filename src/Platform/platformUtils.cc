#include "platformUtils.h"

using namespace torch;

pair<vector<mdlp::labels_t>, map<string, int>> discretize(vector<mdlp::samples_t>& X, mdlp::labels_t& y, vector<string> features)
{
    vector<mdlp::labels_t> Xd;
    map<string, int> maxes;
    auto fimdlp = mdlp::CPPFImdlp();
    for (int i = 0; i < X.size(); i++) {
        fimdlp.fit(X[i], y);
        mdlp::labels_t& xd = fimdlp.transform(X[i]);
        maxes[features[i]] = *max_element(xd.begin(), xd.end()) + 1;
        Xd.push_back(xd);
    }
    return { Xd, maxes };
}

vector<mdlp::labels_t> discretizeDataset(vector<mdlp::samples_t>& X, mdlp::labels_t& y)
{
    vector<mdlp::labels_t> Xd;
    auto fimdlp = mdlp::CPPFImdlp();
    for (int i = 0; i < X.size(); i++) {
        fimdlp.fit(X[i], y);
        mdlp::labels_t& xd = fimdlp.transform(X[i]);
        Xd.push_back(xd);
    }
    return Xd;
}

bool file_exists(const std::string& name)
{
    if (FILE* file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

tuple < Tensor, Tensor, vector<string>> loadDataset(string name, bool discretize)
{
    auto handler = ArffFiles();
    handler.load(PATH + static_cast<string>(name) + ".arff");
    // Get Dataset X, y
    vector<mdlp::samples_t>& X = handler.getX();
    mdlp::labels_t& y = handler.getY();
    // Get className & Features
    auto className = handler.getClassName();
    vector<string> features;
    for (auto feature : handler.getAttributes()) {
        features.push_back(feature.first);
    }
    Tensor Xd;
    if (discretize) {
        auto Xr = discretizeDataset(X, y);
        Xd = torch::zeros({ static_cast<int64_t>(Xr[0].size()), static_cast<int64_t>(Xr.size()) }, torch::kInt64);
        for (int i = 0; i < features.size(); ++i) {
            Xd.index_put_({ "...", i }, torch::tensor(Xr[i], torch::kInt64));
        }
    } else {
        Xd = torch::zeros({ static_cast<int64_t>(X[0].size()), static_cast<int64_t>(X.size()) }, torch::kFloat64);
        for (int i = 0; i < features.size(); ++i) {
            Xd.index_put_({ "...", i }, torch::tensor(X[i], torch::kFloat64));
        }
    }
    return { Xd, torch::tensor(y, torch::kInt64), features };
}

pair <map<string, int>, map<string, vector<int>>> discretize_info(Tensor& X, Tensor& y, vector<string> features, string className)
{
    map<string, int> maxes;
    map<string, vector<int>> states;
    for (int i = 0; i < X.size(1); i++) {
        maxes[features[i]] = X.select(1, i).max().item<int>() + 1;
        states[features[i]] = vector<int>(maxes[features[i]]);
    }
    maxes[className] = y.max().item<int>() + 1;
    states[className] = vector<int>(maxes[className]);
    return { maxes, states };
}

tuple<vector<vector<int>>, vector<int>, vector<string>, string, map<string, vector<int>>> loadFile(string name)
{
    auto handler = ArffFiles();
    handler.load(PATH + static_cast<string>(name) + ".arff");
    // Get Dataset X, y
    vector<mdlp::samples_t>& X = handler.getX();
    mdlp::labels_t& y = handler.getY();
    // Get className & Features
    auto className = handler.getClassName();
    vector<string> features;
    for (auto feature : handler.getAttributes()) {
        features.push_back(feature.first);
    }
    // Discretize Dataset
    vector<mdlp::labels_t> Xd;
    map<string, int> maxes;
    tie(Xd, maxes) = discretize(X, y, features);
    maxes[className] = *max_element(y.begin(), y.end()) + 1;
    map<string, vector<int>> states;
    for (auto feature : features) {
        states[feature] = vector<int>(maxes[feature]);
    }
    states[className] = vector<int>(maxes[className]);
    return { Xd, y, features, className, states };
}