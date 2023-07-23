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
pair<Tensor, map<string, int>> discretizeTorch(Tensor& X, Tensor& y, vector<string> features)
{
    map<string, int> maxes;
    auto fimdlp = mdlp::CPPFImdlp();
    auto Xd = torch::zeros_like(X, torch::kInt64);
    auto yv = vector<int>(y.data_ptr<int>(), y.data_ptr<int>() + y.size(0));
    for (int i = 0; i < X.size(1); i++) {
        auto xv = vector<float>(X.select(1, i).data_ptr<float>(), X.select(1, i).data_ptr<float>() + X.size(0));
        fimdlp.fit(xv, yv);
        auto xdv = fimdlp.transform(xv);
        auto xd = torch::tensor(xdv, torch::kInt64);
        maxes[features[i]] = xd.max().item<int>() + 1;
        Xd.index_put_({ "...", i }, xd);
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

tuple < Tensor, Tensor, vector<string>, string> loadDataset(string name, bool discretize, bool class_last)
{
    auto handler = ArffFiles();
    handler.load(PATH + static_cast<string>(name) + ".arff", class_last);
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
    return { Xd, torch::tensor(y, torch::kInt64), features, className };
}

map<string, vector<int>> get_states(Tensor& X, Tensor& y, vector<string> features, string className)
{
    int max;
    map<string, vector<int>> states;
    for (int i = 0; i < X.size(1); i++) {
        max = X.select(1, i).max().item<int>() + 1;
        states[features[i]] = vector<int>(max);
    }
    max = y.max().item<int>() + 1;
    states[className] = vector<int>(max);
    return states;
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