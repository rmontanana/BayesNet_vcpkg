#ifndef FOLDING_H
#define FOLDING_H
#include <vector>
using namespace std;
class KFold {
private:
    int k;
    int n;
    vector<int> indices;
    int seed;
public:
    KFold(int k, int n, int seed = -1);
    pair<vector<int>, vector<int>> getFold(int nFold);
};
class StratifiedKFold {
private:
    int k;
    int n;
    vector<vector<int>> stratified_indices;
    unsigned seed;
public:
    StratifiedKFold(int k, const vector<int>& y, int seed = -1);
    pair<vector<int>, vector<int>> getFold(int nFold);
};
#endif