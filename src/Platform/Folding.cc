#include "Folding.h"
#include <algorithm>
#include <random>

using namespace std;

KFold::KFold(int k, int n, int seed)
{
    this->k = k;
    this->n = n;
    indices = vector<int>(n);
    iota(begin(indices), end(indices), 0); // fill with 0, 1, ..., n - 1
    shuffle(indices.begin(), indices.end(), default_random_engine(seed));
}
pair<vector<int>, vector<int>> KFold::getFold(int nFold)
{
    if (nFold >= k || nFold < 0) {
        throw invalid_argument("nFold (" + to_string(nFold) + ") must be less than k (" + to_string(k) + ")");
    }
    int nTest = n / k;
    auto train = vector<int>();
    auto test = vector<int>();
    for (int i = 0; i < n; i++) {
        if (i >= nTest * nFold && i < nTest * (nFold + 1)) {
            test.push_back(indices[i]);
        } else {
            train.push_back(indices[i]);
        }
    }
    return { train, test };
}