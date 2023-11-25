#ifndef GRIDSEARCH_H
#define GRIDSEARCH_H
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "Datasets.h"
#include "HyperParameters.h"
#include "GridData.h"

namespace platform {
    using json = nlohmann::json;
    struct ConfigGrid {
        std::string model;
        std::string score;
        std::string path;
        std::string input_file;
        std::string output_file;
        bool quiet;
        bool discretize;
        bool stratified;
        int n_folds;
        std::vector<int> seeds;
    };
    class GridSearch {
    public:
        explicit GridSearch(struct ConfigGrid& config);
        void go();
        void save() const;
        ~GridSearch() = default;
    private:
        double processFile(std::string fileName, Datasets& datasets, HyperParameters& hyperparameters);
        json results;
        struct ConfigGrid config;
    };
} /* namespace platform */
#endif /* GRIDSEARCH_H */