#ifndef FOLDING_H
#define FOLDING_H
#include <vector>
using namespace std;
class KFold {
private:
    int k;
    int n;
    int seed;
    vector<int> indices;
public:
    KFold(int k, int n, int seed = -1);
    pair<vector<int>, vector<int>> getFold(int nFold);
};
class StratifiedKFold {
private:
    int k;
    int n;
    int seed;
    vector<vector<int>> stratified_indices;
public:
    StratifiedKFold(int k, const vector<int>& y, int seed = -1);
    pair<vector<int>, vector<int>> getFold(int nFold);
};
#endif