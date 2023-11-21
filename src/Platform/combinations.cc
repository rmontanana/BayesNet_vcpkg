#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

json generateCombinations(json::iterator index, const json::iterator last, std::vector<json>& output, json currentCombination)
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

int main()
{
    json input = R"(
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
    auto output = std::vector<json>();
    for (json line : input) {
        generateCombinations(line.begin(), line.end(), output, json({}));
    }
    // Print the generated combinations
    int i = 0;
    for (const auto& item : output) {
        std::cout << i++ << " " << item.dump() << std::endl;
    }
    return 0;
}
