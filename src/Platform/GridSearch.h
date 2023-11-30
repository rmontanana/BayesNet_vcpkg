#ifndef GRIDSEARCH_H
#define GRIDSEARCH_H
#include <string>
#include <map>
#include <nlohmann/json.hpp>
#include "Datasets.h"
#include "HyperParameters.h"
#include "GridData.h"

namespace platform {
    using json = nlohmann::json;
    struct ConfigGrid {
        std::string model;
        std::string score;
        std::string continue_from;
        bool quiet;
        bool only; // used with continue_from to only compute that dataset
        bool discretize;
        bool stratified;
        int n_folds;
        std::vector<int> seeds;
    };
    class GridSearch {
    public:
        explicit GridSearch(struct ConfigGrid& config);
        void go();
        ~GridSearch() = default;
        json getResults();
    private:
        void save(json& results) const;
        vector<std::string> processDatasets(Datasets& datasets);
        double processFile(std::string fileName, Datasets& datasets, HyperParameters& hyperparameters);
        struct ConfigGrid config;
    };
} /* namespace platform */
#endif /* GRIDSEARCH_H */