#ifndef FOLDING_H
#define FOLDING_H
#include <vector>
using namespace std;
class KFold {
private:
    int k;
    int n;
    vector<int> indices;

public:
    KFold(int k, int n, int seed);
    pair<vector<int>, vector<int>> getFold(int);
};
class KStratifiedFold {

};
#endif