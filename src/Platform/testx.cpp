#include "Folding.h"
#include <torch/torch.h>
#include "nlohmann/json.hpp"
#include "map"
#include <iostream>
#include <sstream>
#include "Datasets.h"
#include "Network.h"
#include "ArffFiles.h"
#include "CPPFImdlp.h"
#include "CFS.h"
#include "IWSS.h"
#include "FCBF.h"

using namespace std;
using namespace platform;
using namespace torch;

string counts(vector<int> y, vector<int> indices)
{
    auto result = map<int, int>();
    stringstream oss;
    for (auto i = 0; i < indices.size(); ++i) {
        result[y[indices[i]]]++;
    }
    string final_result = "";
    for (auto i = 0; i < result.size(); ++i)
        oss << i << " -> " << setprecision(2) << fixed
        << (double)result[i] * 100 / indices.size() << "% (" << result[i] << ") //";
    oss << endl;
    return oss.str();
}
class Paths {
public:
    static string datasets()
    {
        return "datasets/";
    }
};

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

bool file_exists(const string& name)
{
    if (FILE* file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

tuple<Tensor, Tensor, vector<string>, string, map<string, vector<int>>> loadDataset(const string& name, bool class_last, bool discretize_dataset)
{
    auto handler = ArffFiles();
    handler.load(Paths::datasets() + static_cast<string>(name) + ".arff", class_last);
    // Get Dataset X, y
    vector<mdlp::samples_t>& X = handler.getX();
    mdlp::labels_t& y = handler.getY();
    // Get className & Features
    auto className = handler.getClassName();
    vector<string> features;
    auto attributes = handler.getAttributes();
    transform(attributes.begin(), attributes.end(), back_inserter(features), [](const auto& pair) { return pair.first; });
    Tensor Xd;
    auto states = map<string, vector<int>>();
    if (discretize_dataset) {
        auto Xr = discretizeDataset(X, y);
        Xd = torch::zeros({ static_cast<int>(Xr.size()), static_cast<int>(Xr[0].size()) }, torch::kInt32);
        for (int i = 0; i < features.size(); ++i) {
            states[features[i]] = vector<int>(*max_element(Xr[i].begin(), Xr[i].end()) + 1);
            auto item = states.at(features[i]);
            iota(begin(item), end(item), 0);
            Xd.index_put_({ i, "..." }, torch::tensor(Xr[i], torch::kInt32));
        }
        states[className] = vector<int>(*max_element(y.begin(), y.end()) + 1);
        iota(begin(states.at(className)), end(states.at(className)), 0);
    } else {
        Xd = torch::zeros({ static_cast<int>(X.size()), static_cast<int>(X[0].size()) }, torch::kFloat32);
        for (int i = 0; i < features.size(); ++i) {
            Xd.index_put_({ i, "..." }, torch::tensor(X[i]));
        }
    }
    return { Xd, torch::tensor(y, torch::kInt32), features, className, states };
}

tuple<vector<vector<int>>, vector<int>, vector<string>, string, map<string, vector<int>>> loadFile(const string& name)
{
    auto handler = ArffFiles();
    handler.load(Paths::datasets() + static_cast<string>(name) + ".arff");
    // Get Dataset X, y
    vector<mdlp::samples_t>& X = handler.getX();
    mdlp::labels_t& y = handler.getY();
    // Get className & Features
    auto className = handler.getClassName();
    vector<string> features;
    auto attributes = handler.getAttributes();
    transform(attributes.begin(), attributes.end(), back_inserter(features), [](const auto& pair) { return pair.first; });
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
class RawDatasets {
public:
    RawDatasets(const string& file_name, bool discretize)
    {
        // Xt can be either discretized or not
        tie(Xt, yt, featurest, classNamet, statest) = loadDataset(file_name, true, discretize);
        // Xv is always discretized
        tie(Xv, yv, featuresv, classNamev, statesv) = loadFile(file_name);
        auto yresized = torch::transpose(yt.view({ yt.size(0), 1 }), 0, 1);
        dataset = torch::cat({ Xt, yresized }, 0);
        nSamples = dataset.size(1);
        weights = torch::full({ nSamples }, 1.0 / nSamples, torch::kDouble);
        weightsv = vector<double>(nSamples, 1.0 / nSamples);
        classNumStates = discretize ? statest.at(classNamet).size() : 0;
    }
    torch::Tensor Xt, yt, dataset, weights;
    vector<vector<int>> Xv;
    vector<double> weightsv;
    vector<int> yv;
    vector<string> featurest, featuresv;
    map<string, vector<int>> statest, statesv;
    string classNamet, classNamev;
    int nSamples, classNumStates;
    double epsilon = 1e-5;
};
int main()
{
    // map<string, string> balance = {
    //     {"iris", "33,33% (50) / 33,33% (50) / 33,33% (50)"},
    //     {"diabetes", "34,90% (268) / 65,10% (500)"},
    //     {"ecoli", "42,56% (143) / 22,92% (77) / 0,60% (2) / 0,60% (2) / 10,42% (35) / 5,95% (20) / 1,49% (5) / 15,48% (52)"},
    //     {"glass", "32,71% (70) / 7,94% (17) / 4,21% (9) / 35,51% (76) / 13,55% (29) / 6,07% (13)"}
    // };
    // for (const auto& file_name : { "iris", "glass", "ecoli", "diabetes" }) {
    //     auto dt = Datasets(true, "Arff");
    //     auto [X, y] = dt.getVectors(file_name);
    //     //auto fold = KFold(5, 150);
    //     auto fold = StratifiedKFold(5, y, -1);
    //     cout << "***********************************************************************************************" << endl;
    //     cout << "Dataset: " << file_name << endl;
    //     cout << "Nº Samples: " << dt.getNSamples(file_name) << endl;
    //     cout << "Class states: " << dt.getNClasses(file_name) << endl;
    //     cout << "Balance: " << balance.at(file_name) << endl;
    //     for (int i = 0; i < 5; ++i) {
    //         cout << "Fold: " << i << endl;
    //         auto [train, test] = fold.getFold(i);
    //         cout << "Train: ";
    //         cout << "(" << train.size() << "): ";
    //         // for (auto j = 0; j < static_cast<int>(train.size()); j++)
    //         //     cout << train[j] << ", ";
    //         cout << endl;
    //         cout << "Train Statistics : " << counts(y, train);
    //         cout << "-------------------------------------------------------------------------------" << endl;
    //         cout << "Test: ";
    //         cout << "(" << test.size() << "): ";
    //         // for (auto j = 0; j < static_cast<int>(test.size()); j++)
    //         //     cout << test[j] << ", ";
    //         cout << endl;
    //         cout << "Test Statistics: " << counts(y, test);
    //         cout << "==============================================================================" << endl;
    //     }
    //     cout << "***********************************************************************************************" << endl;
    // }
    // const string file_name = "iris";
    // auto net = bayesnet::Network();
    // auto dt = Datasets(true, "Arff");
    // auto raw = RawDatasets("iris", true);
    // auto [X, y] = dt.getVectors(file_name);
    // cout << "Dataset dims " << raw.dataset.sizes() << endl;
    // cout << "weights dims " << raw.weights.sizes() << endl;
    // cout << "States dims " << raw.statest.size() << endl;
    // cout << "features: ";
    // for (const auto& feature : raw.featurest) {
    //     cout << feature << ", ";
    //     net.addNode(feature);
    // }
    // net.addNode(raw.classNamet);
    // cout << endl;
    // net.fit(raw.dataset, raw.weights, raw.featurest, raw.classNamet, raw.statest);
    auto dt = Datasets(true, "Arff");
    nlohmann::json output;
    for (const auto& name : dt.getNames()) {
        // for (const auto& name : { "iris" }) {
        auto [X, y] = dt.getTensors(name);
        auto features = dt.getFeatures(name);
        auto states = dt.getStates(name);
        auto className = dt.getClassName(name);
        int maxFeatures = 0;
        auto classNumStates = states.at(className).size();
        torch::Tensor weights = torch::full({ X.size(1) }, 1.0 / X.size(1), torch::kDouble);
        auto dataset = X;
        auto yresized = torch::transpose(y.view({ y.size(0), 1 }), 0, 1);
        dataset = torch::cat({ dataset, yresized }, 0);
        auto cfs = bayesnet::CFS(dataset, features, className, maxFeatures, classNumStates, weights);
        auto fcbf = bayesnet::FCBF(dataset, features, className, maxFeatures, classNumStates, weights, 1e-7);
        auto iwss = bayesnet::IWSS(dataset, features, className, maxFeatures, classNumStates, weights, 0.5);
        cout << "Dataset: " << setw(20) << name << flush;
        cfs.fit();
        cout << " CFS: " << setw(4) << cfs.getFeatures().size() << flush;
        fcbf.fit();
        cout << " FCBF: " << setw(4) << fcbf.getFeatures().size() << flush;
        iwss.fit();
        cout << " IWSS: " << setw(4) << iwss.getFeatures().size() << flush;
        cout << endl;
        output[name]["CFS"] = cfs.getFeatures();
        output[name]["FCBF"] = fcbf.getFeatures();
        output[name]["IWSS"] = iwss.getFeatures();
    }
    ofstream file("features_cpp.json");
    file << output;
    file.close();

}

