#ifndef GRIDDATA_H
#define GRIDDATA_H
#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

namespace platform {
    using json = nlohmann::json;
    class GridData {
    public:
        explicit GridData(const std::string& fileName);
        ~GridData() = default;
        std::vector<json> getGrid();
        int getNumCombinations();
    private:
        json generateCombinations(json::iterator index, const json::iterator last, std::vector<json>& output, json currentCombination);
        int computeNumCombinations(const json& line);
        json grid;
    };
} /* namespace platform */
#endif /* GRIDDATA_H */