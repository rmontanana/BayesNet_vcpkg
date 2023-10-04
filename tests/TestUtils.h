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
#endif //TEST_UTILS_H

#