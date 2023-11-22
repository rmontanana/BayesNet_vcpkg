#ifndef GRIDSEARCH_H
#define GRIDSEARCH_H
#include <string>
#include <vector>
#include "Datasets.h"
#include "HyperParameters.h"
#include "GridData.h"

namespace platform {
    struct ConfigGrid {
        std::string model;
        std::string score;
        std::string path;
        std::string input_file;
        std::string output_file;
        bool discretize;
        bool stratified;
        int n_folds;
        std::vector<int> seeds;
    };
    class GridSearch {
    public:
        explicit GridSearch(struct ConfigGrid& config);
        void go();
        void save();
        ~GridSearch() = default;
    private:
        void processFile(std::string fileName, Datasets& datasets, HyperParameters& hyperparameters);
        struct ConfigGrid config;
        GridData grid;
    };
} /* namespace platform */
#endif /* GRIDSEARCH_H */