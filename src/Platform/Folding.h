#ifndef FOLDING_H
#define FOLDING_H
#include <torch/torch.h>
#include <vector>
#include <random> 
namespace platform {
    class Fold {
    protected:
        int k;
        int n;
        int seed;
        std::default_random_engine random_seed;
    public:
        Fold(int k, int n, int seed = -1);
        virtual std::pair<std::vector<int>, std::vector<int>> getFold(int nFold) = 0;
        virtual ~Fold() = default;
        int getNumberOfFolds() { return k; }
    };
    class KFold : public Fold {
    private:
        std::vector<int> indices;
    public:
        KFold(int k, int n, int seed = -1);
        std::pair<std::vector<int>, std::vector<int>> getFold(int nFold) override;
    };
    class StratifiedKFold : public Fold {
    private:
        std::vector<int> y;
        std::vector<std::vector<int>> stratified_indices;
        void build();
        bool faulty = false; // Only true if the number of samples of any class is less than the number of folds.
    public:
        StratifiedKFold(int k, const std::vector<int>& y, int seed = -1);
        StratifiedKFold(int k, torch::Tensor& y, int seed = -1);
        std::pair<std::vector<int>, std::vector<int>> getFold(int nFold) override;
        bool isFaulty() { return faulty; }
    };
}
#endif