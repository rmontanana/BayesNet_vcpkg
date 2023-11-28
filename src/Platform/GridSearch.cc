#include <iostream>
#include <torch/torch.h>
#include "GridSearch.h"
#include "Models.h"
#include "Paths.h"
#include "Folding.h"
#include "Colors.h"

namespace platform {
    std::string get_date()
    {
        time_t rawtime;
        tm* timeinfo;
        time(&rawtime);
        timeinfo = std::localtime(&rawtime);
        std::ostringstream oss;
        oss << std::put_time(timeinfo, "%Y-%m-%d");
        return oss.str();
    }
    std::string get_time()
    {
        time_t rawtime;
        tm* timeinfo;
        time(&rawtime);
        timeinfo = std::localtime(&rawtime);
        std::ostringstream oss;
        oss << std::put_time(timeinfo, "%H:%M:%S");
        return oss.str();
    }
    GridSearch::GridSearch(struct ConfigGrid& config) : config(config)
    {
        this->config.output_file = config.path + "grid_" + config.model + "_output.json";
        this->config.input_file = config.path + "grid_" + config.model + "_input.json";
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
        auto className = datasets.getClassName(fileName);
        double totalScore = 0.0;
        int numItems = 0;
        for (const auto& seed : config.seeds) {
            if (!config.quiet)
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
                if (!config.quiet)
                    showProgressFold(nfold + 1, getColor(clf->getStatus()), "a");
                clf->fit(X_train, y_train, features, className, states);
                // Test model
                if (!config.quiet)
                    showProgressFold(nfold + 1, getColor(clf->getStatus()), "b");
                totalScore += clf->score(X_test, y_test);
                numItems++;
                if (!config.quiet)
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
        // Load previous results
        json results;
        auto datasets_names = datasets.getNames();
        if (config.continue_from != "No") {
            // Continue previous execution:
            // Load previous results & remove datasets already processed
            if (std::find(datasets_names.begin(), datasets_names.end(), config.continue_from) == datasets_names.end()) {
                throw std::invalid_argument("Dataset " + config.continue_from + " not found");
            }
            if (!config.quiet)
                std::cout << "* Loading previous results" << std::endl;
            try {
                std::ifstream file(config.output_file);
                if (file.is_open()) {
                    results = json::parse(file);
                }
            }
            catch (const std::exception& e) {
                std::cerr << "* There were no previous results" << std::endl;
                std::cerr << "* Initizalizing new results" << std::endl;
                results = json();
            }
            // Remove datasets already processed
            vector< string >::iterator it = datasets_names.begin();
            while (it != datasets_names.end()) {
                if (*it != config.continue_from) {
                    it = datasets_names.erase(it);
                } else {
                    if (config.only)
                        ++it;
                    else
                        break;
                }
            }
        }
        // Create model
        std::cout << "***************** Starting Gridsearch *****************" << std::endl;
        std::cout << "input file=" << config.input_file << std::endl;
        auto grid = GridData(config.input_file);
        auto totalComb = grid.getNumCombinations();
        std::cout << "* Doing " << totalComb << " combinations for each dataset/seed/fold" << std::endl;
        // Generate hyperparameters grid & run gridsearch
        // Check each combination of hyperparameters for each dataset and each seed
        for (const auto& dataset : datasets_names) {
            if (!config.quiet)
                std::cout << "- " << setw(20) << left << dataset << " " << right << flush;
            int num = 0;
            double bestScore = 0.0;
            json bestHyperparameters;
            auto combinations = grid.getGrid();
            for (const auto& hyperparam_line : combinations) {
                if (!config.quiet)
                    showProgressComb(++num, totalComb, Colors::CYAN());
                auto hyperparameters = platform::HyperParameters(datasets.getNames(), hyperparam_line);
                double score = processFile(dataset, datasets, hyperparameters);
                if (score > bestScore) {
                    bestScore = score;
                    bestHyperparameters = hyperparam_line;
                }
            }
            if (!config.quiet) {
                std::cout << "end." << " Score: " << setw(9) << setprecision(7) << fixed
                    << bestScore << " [" << bestHyperparameters.dump() << "]" << std::endl;
            }
            results[dataset]["score"] = bestScore;
            results[dataset]["hyperparameters"] = bestHyperparameters;
            results[dataset]["date"] = get_date() + " " + get_time();
            results[dataset]["grid"] = grid.getInputGrid();
            // Save partial results
            save(results);
        }
        // Save final results
        save(results);
        std::cout << "***************** Ending Gridsearch *******************" << std::endl;
    }
    void GridSearch::save(json& results) const
    {
        std::ofstream file(config.output_file);
        file << results.dump(4);
    }
} /* namespace platform */