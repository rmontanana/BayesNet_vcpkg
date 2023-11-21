#include <iostream>
#include "GridSearch.h"
#include "Paths.h"
#include "Datasets.h"
#include "HyperParameters.h"

namespace platform {
    GridSearch::GridSearch(struct ConfigGrid& config) : config(config)
    {
        this->config.output_file = config.path + "grid_" + config.model + "_output.json";
    }
    void GridSearch::go()
    {
        // Load datasets
        auto datasets = platform::Datasets(config.discretize, Paths::datasets());
        int i = 0;
        for (const auto& item : grid.getGrid("BoostAODE")) {
            std::cout << i++ << " hyperparams: " << item.dump() << std::endl;
        }
        // Load hyperparameters
        // auto hyperparameters = platform::HyperParameters(datasets.getNames(), config.input_file);
        // Check if hyperparameters are valid
        // auto valid_hyperparameters = platform::Models::instance()->getHyperparameters(config.model);
        // hyperparameters.check(valid_hyperparameters, config.model);
        // // Load model
        // auto model = platform::Models::instance()->get(config.model);
        // // Run gridsearch
        // auto grid = platform::Grid(datasets, hyperparameters, model, config.score, config.discretize, config.stratified, config.n_folds, config.seeds);
        // grid.run();
        // // Save results
        // grid.save(config.output_file);
    }
    void GridSearch::save()
    {

    }

} /* namespace platform */