#include "Folding.h"
#include <algorithm>
#include <map>
#include <random>

using namespace std;

KFold::KFold(int k, int n, int seed) : k(k), n(n), seed(seed)
{
    indices = vector<int>(n);
    iota(begin(indices), end(indices), 0); // fill with 0, 1, ..., n - 1
    random_device rd;
    default_random_engine random_seed(seed == -1 ? rd() : seed);
    shuffle(indices.begin(), indices.end(), random_seed);
}
pair<vector<int>, vector<int>> KFold::getFold(int nFold)
{

    if (nFold >= k || nFold < 0) {
        throw out_of_range("nFold (" + to_string(nFold) + ") must be less than k (" + to_string(k) + ")");
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
StratifiedKFold::StratifiedKFold(int k, const vector<int>& y, int seed) :
    k(k), seed(seed)
{
    // n = y.size();
    // map<int, vector<int>> class_to_indices;
    // for (int i = 0; i < n; ++i) {
    //     class_to_indices[y[i]].push_back(i);
    // }
    // random_device rd;
    // default_random_engine random_seed(seed == -1 ? rd() : seed);
    // for (auto& [cls, indices] : class_to_indices) {
    //     shuffle(indices.begin(), indices.end(), random_seed);
    //     int fold_size = n / k;
    //     for (int i = 0; i < k; ++i) {
    //         int start = i * fold_size;
    //         int end = (i == k - 1) ? indices.size() : (i + 1) * fold_size;
    //         stratified_indices.emplace_back(indices.begin() + start, indices.begin() + end);
    //     }
    // }
    n = y.size();
    stratified_indices.resize(k);
    vector<int> class_counts(*max_element(y.begin(), y.end()) + 1, 0);
    for (auto i = 0; i < n; ++i) {
        class_counts[y[i]]++;
    }
    vector<int> class_starts(class_counts.size());
    partial_sum(class_counts.begin(), class_counts.end() - 1, class_starts.begin() + 1);
    vector<int> indices(n);
    for (auto i = 0; i < n; ++i) {
        int label = y[i];
        stratified_indices[class_starts[label]] = i;
        class_starts[label]++;
    }
    int fold_size = n / k;
    int remainder = n % k;
    int start = 0;
    for (auto i = 0; i < k; ++i) {
        int fold_length = fold_size + (i < remainder ? 1 : 0);
        stratified_indices[i].resize(fold_length);
        copy(indices.begin() + start, indices.begin() + start + fold_length, stratified_indices[i].begin());
        start += fold_length;
    }
}
pair<vector<int>, vector<int>> StratifiedKFold::getFold(int nFold)
{
    if (nFold >= k || nFold < 0) {
        throw out_of_range("nFold (" + to_string(nFold) + ") must be less than k (" + to_string(k) + ")");
    }
    vector<int> test_indices = stratified_indices[nFold];
    vector<int> train_indices;
    for (int i = 0; i < k; ++i) {
        if (i == nFold) continue;
        train_indices.insert(train_indices.end(), stratified_indices[i].begin(), stratified_indices[i].end());
    }
    return { train_indices, test_indices };
}