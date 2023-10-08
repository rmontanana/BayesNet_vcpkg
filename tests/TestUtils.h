#ifndef TEST_UTILS_H
#define TEST_UTILS_H
#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include "ArffFiles.h"
#include "CPPFImdlp.h"
using namespace std;

bool file_exists(const std::string& name);
pair<vector<mdlp::labels_t>, map<string, int>> discretize(vector<mdlp::samples_t>& X, mdlp::labels_t& y, vector<string> features);
vector<mdlp::labels_t> discretizeDataset(vector<mdlp::samples_t>& X, mdlp::labels_t& y);
tuple<vector<vector<int>>, vector<int>, vector<string>, string, map<string, vector<int>>> loadFile(const string& name);
tuple<torch::Tensor, torch::Tensor, vector<string>, string, map<string, vector<int>>> loadDataset(const string& name, bool class_last, bool discretize_dataset);

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

#endif //TEST_UTILS_H