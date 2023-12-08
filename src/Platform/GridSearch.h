#ifndef GRIDSEARCH_H
#define GRIDSEARCH_H
#include <string>
#include <map>
#include <nlohmann/json.hpp>
#include "Datasets.h"
#include "HyperParameters.h"
#include "GridData.h"
#include "Timer.h"

namespace platform {
    using json = nlohmann::json;
    struct ConfigGrid {
        std::string model;
        std::string score;
        std::string continue_from;
        std::string platform;
        bool quiet;
        bool only; // used with continue_from to only compute that dataset
        bool discretize;
        bool stratified;
        int nested;
        int n_folds;
        json excluded;
        std::vector<int> seeds;
    };
    class GridSearch {
    public:
        explicit GridSearch(struct ConfigGrid& config);
        void go();
        ~GridSearch() = default;
        json getResults();
        static inline std::string NO_CONTINUE() { return "NO_CONTINUE"; }
    private:
        void save(json& results);
        json initializeResults();
        vector<std::string> processDatasets(Datasets& datasets);
        pair<double, json> processFileSingle(std::string fileName, Datasets& datasets, std::vector<json>& combinations);
        pair<double, json> processFileNested(std::string fileName, Datasets& datasets, std::vector<json>& combinations);
        struct ConfigGrid config;
        Timer timer; // used to measure the time of the whole process
    };
} /* namespace platform */
#endif /* GRIDSEARCH_H */