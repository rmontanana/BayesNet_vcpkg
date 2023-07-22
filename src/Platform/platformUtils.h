#ifndef PLATFORM_UTILS_H
#define PLATFORM_UTILS_H
#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include "ArffFiles.h"
#include "CPPFImdlp.h"
using namespace std;
const string PATH = "../../data/";

bool file_exists(const std::string& name);
pair<vector<mdlp::labels_t>, map<string, int>> discretize(vector<mdlp::samples_t>& X, mdlp::labels_t& y, vector<string> features);
tuple<vector<vector<int>>, vector<int>, vector<string>, string, map<string, vector<int>>> loadFile(string name);
tuple<torch::Tensor, torch::Tensor, vector<string>> loadDataset(string name, bool discretize);
pair <map<string, int>, map<string, vector<int>>> discretize_info(torch::Tensor& X, torch::Tensor& y);
#endif //PLATFORM_UTILS_H
