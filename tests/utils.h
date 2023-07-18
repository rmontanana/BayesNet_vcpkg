#include <string>
#include <vector>
#include <map>
#include <tuple>
#include "../sample/ArffFiles.h"
#include "../sample/CPPFImdlp.h"
#ifndef BAYESNET_UTILS_H
#define BAYESNET_UTILS_H
using namespace std;
const string PATH = "../../data/";
pair<vector<mdlp::labels_t>, map<string, int>> discretize(vector<mdlp::samples_t> &X, mdlp::labels_t &y, vector<string> features);
tuple<vector<vector<int>>, vector<int>, vector<string>, string, map<string, vector<int>>> loadFile(string name);
#endif //BAYESNET_UTILS_H
