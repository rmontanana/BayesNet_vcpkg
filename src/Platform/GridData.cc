#include "GridData.h"
#include <fstream>

namespace platform {
    GridData::GridData(const std::string& fileName)
    {
        std::ifstream resultData(fileName);
        if (resultData.is_open()) {
            grid = json::parse(resultData);
        } else {
            throw std::invalid_argument("Unable to open input file. [" + fileName + "]");
        }
    }
    int GridData::computeNumCombinations(const json& line)
    {
        int numCombinations = 1;
        for (const auto& item : line.items()) {
            numCombinations *= item.value().size();
        }
        return numCombinations;
    }
    int GridData::getNumCombinations()
    {
        int numCombinations = 0;
        for (const auto& line : grid) {
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
    std::vector<json> GridData::getGrid()
    {
        auto result = std::vector<json>();
        for (json line : grid) {
            generateCombinations(line.begin(), line.end(), result, json({}));
        }
        return result;
    }
} /* namespace platform */