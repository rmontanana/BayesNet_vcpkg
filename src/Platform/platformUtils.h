#ifndef PLATFORM_UTILS_H
#define PLATFORM_UTILS_H
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
#endif //PLATFORM_UTILS_H
