#include "GridData.h"
#include <iostream>

namespace platform {
    GridData::GridData()
    {
        auto boostaode = R"(
            [
                {
                    "convergence": [true, false],
                    "ascending": [true, false],
                    "repeatSparent": [true, false],
                    "select_features": ["CFS", "FCBF"],
                    "tolerance": [0, 3, 5],
                    "threshold": [1e-7]
                },
                {
                    "convergence": [true, false],
                    "ascending": [true, false],
                    "repeatSparent": [true, false],
                    "select_features": ["IWSS"],
                    "tolerance": [0, 3, 5],
                    "threshold": [0.5]
                
                }
            ]
        )"_json;
        grid["BoostAODE"] = boostaode;
    }
    int GridData::computeNumCombinations(const json& line)
    {
        int numCombinations = 1;
        for (const auto& item : line) {
            for (const auto& hyperparam : item.items()) {
                numCombinations *= item.size();
            }
        }
        return numCombinations;
    }
    std::vector<json> GridData::doCombination(const std::string& model)
    {
        int numTotal = 0;
        for (const auto& item : grid[model]) {
            numTotal += computeNumCombinations(item);
        }
        auto result = std::vector<json>(numTotal);
        int base = 0;
        for (const auto& item : grid[model]) {
            int numCombinations = computeNumCombinations(item);
            int line = 0;
            for (const auto& hyperparam : item.items()) {
                int numValues = hyperparam.value().size();
                for (const auto& value : hyperparam.value()) {
                    for (int i = 0; i < numCombinations / numValues; i++) {
                        result[base + line++][hyperparam.key()] = value;
                        //std::cout << "line=" << base + line << " " << hyperparam.key() << "=" << value << std::endl;
                    }
                }
            }
            base += numCombinations;
        }
        for (const auto& item : result) {
            std::cout << item.dump() << std::endl;
        }
        return result;
    }
} /* namespace platform */