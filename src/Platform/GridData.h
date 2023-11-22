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
        GridData();
        ~GridData() = default;
        std::vector<json> getGrid(const std::string& model);
        int getNumCombinations(const std::string& model);
    private:
        json generateCombinations(json::iterator index, const json::iterator last, std::vector<json>& output, json currentCombination);
        int computeNumCombinations(const json& line);
        std::map<std::string, json> grid;
    };
} /* namespace platform */
#endif /* GRIDDATA_H */