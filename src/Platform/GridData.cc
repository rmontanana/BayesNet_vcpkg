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
        for (const auto& item : line.items()) {
            numCombinations *= item.value().size();
        }
        return numCombinations;
    }
    int GridData::getNumCombinations(const std::string& model)
    {
        int numCombinations = 0;
        for (const auto& line : grid.at(model)) {
            numCombinations += computeNumCombinations(line);
        }
        return numCombinations;
    }
    json GridData::generateCombinations(json::iterator index, const json::iterator last, std::vector<json>& output, json currentCombination)
    {
        if (index == last) {
            // If we reached the end of input, store the current combination
            output.push_back(currentCombination);
            return  currentCombination;
        }
        const auto& key = index.key();
        const auto& values = index.value();
        for (const auto& value : values) {
            auto combination = currentCombination;
            combination[key] = value;
            json::iterator nextIndex = index;
            generateCombinations(++nextIndex, last, output, combination);
        }
        return currentCombination;
    }
    std::vector<json> GridData::getGrid(const std::string& model)
    {
        auto result = std::vector<json>();
        for (json line : grid.at(model)) {
            generateCombinations(line.begin(), line.end(), result, json({}));
        }
        return result;
    }
} /* namespace platform */