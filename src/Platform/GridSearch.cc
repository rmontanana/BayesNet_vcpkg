#include <iostream>
#include <torch/torch.h>
#include "GridSearch.h"
#include "Models.h"
#include "Paths.h"
#include "Folding.h"
#include "Colors.h"

namespace platform {
    GridSearch::GridSearch(struct ConfigGrid& config) : config(config)
    {
        this->config.output_file = config.path + "grid_" + config.model + "_output.json";
    }
    void showProgressComb(const int num, const int total, const std::string& color)
    {
        int spaces = int(log(total) / log(10)) + 1;
        int magic = 37 + 2 * spaces;
        std::string prefix = num == 1 ? "" : string(magic, '\b') + string(magic + 1, ' ') + string(magic + 1, '\b');
        std::cout << prefix << color << "(" << setw(spaces) << num << "/" << setw(spaces) << total << ") " << Colors::RESET() << flush;
    }
    void showProgressFold(int fold, const std::string& color, const std::string& phase)
    {
        std::string prefix = phase == "a" ? "" : "\b\b\b\b";
        std::cout << prefix << color << fold << Colors::RESET() << "(" << color << phase << Colors::RESET() << ")" << flush;
    }
    std::string getColor(bayesnet::status_t status)
    {
        switch (status) {
            case bayesnet::NORMAL:
                return Colors::GREEN();
            case bayesnet::WARNING:
                return Colors::YELLOW();
            case bayesnet::ERROR:
                return Colors::RED();
            default:
                return Colors::RESET();
        }
    }
    double GridSearch::processFile(std::string fileName, Datasets& datasets, HyperParameters& hyperparameters)
    {
        // Get dataset
        auto [X, y] = datasets.getTensors(fileName);
        auto states = datasets.getStates(fileName);
        auto features = datasets.getFeatures(fileName);
        auto samples = datasets.getNSamples(fileName);
        auto className = datasets.getClassName(fileName);
        double totalScore = 0.0;
        int numItems = 0;
        for (const auto& seed : config.seeds) {
            std::cout << "(" << seed << ") doing Fold: " << flush;
            Fold* fold;
            if (config.stratified)
                fold = new StratifiedKFold(config.n_folds, y, seed);
            else
                fold = new KFold(config.n_folds, y.size(0), seed);
            double bestScore = 0.0;
            for (int nfold = 0; nfold < config.n_folds; nfold++) {
                auto clf = Models::instance()->create(config.model);
                clf->setHyperparameters(hyperparameters.get(fileName));
                auto [train, test] = fold->getFold(nfold);
                auto train_t = torch::tensor(train);
                auto test_t = torch::tensor(test);
                auto X_train = X.index({ "...", train_t });
                auto y_train = y.index({ train_t });
                auto X_test = X.index({ "...", test_t });
                auto y_test = y.index({ test_t });
                // Train model
                clf->fit(X_train, y_train, features, className, states);
                showProgressFold(nfold + 1, getColor(clf->getStatus()), "a");
                showProgressFold(nfold + 1, getColor(clf->getStatus()), "b");
                totalScore += clf->score(X_test, y_test);
                numItems++;
                showProgressFold(nfold + 1, getColor(clf->getStatus()), "c");
                std::cout << "\b\b\b,  \b" << flush;
            }
            delete fold;
        }
        return numItems == 0 ? 0.0 : totalScore / numItems;
    }
    void GridSearch::go()
    {
        // Load datasets
        auto datasets = Datasets(config.discretize, Paths::datasets());
        // Create model
        std::cout << "***************** Starting Gridsearch *****************" << std::endl;
        auto totalComb = grid.getNumCombinations(config.model);
        std::cout << "* Doing " << totalComb << " combinations for each dataset/seed/fold" << std::endl;
        // Generate hyperparameters grid & run gridsearch
        // Check each combination of hyperparameters for each dataset and each seed
        for (const auto& dataset : datasets.getNames()) {
            std::cout << "- " << setw(20) << left << dataset << " " << right << flush;
            int num = 0;
            double bestScore = 0.0;
            json bestHyperparameters;
            for (const auto& hyperparam_line : grid.getGrid(config.model)) {
                showProgressComb(++num, totalComb, Colors::CYAN());
                auto hyperparameters = platform::HyperParameters(datasets.getNames(), hyperparam_line);
                double score = processFile(dataset, datasets, hyperparameters);
                if (score > bestScore) {
                    bestScore = score;
                    bestHyperparameters = hyperparam_line;
                }
            }
            std::cout << "end." << " Score: " << setw(9) << setprecision(7) << fixed
                << bestScore << " [" << bestHyperparameters.dump() << "]" << std::endl;
            results[dataset]["score"] = bestScore;
            results[dataset]["hyperparameters"] = bestHyperparameters;
        }
        // Save results
        save();
    }
    void GridSearch::save()
    {
        std::ofstream file(config.output_file);
        file << results.dump(4);
        file.close();
    }
} /* namespace platform */