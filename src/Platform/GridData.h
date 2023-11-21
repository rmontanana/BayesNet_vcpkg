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
        std::vector<json> getGrid(const std::string& model) { return doCombination(model); }
    private:
        int computeNumCombinations(const json& line);
        std::vector<json> doCombination(const std::string& model);
        std::map<std::string, json> grid;
    };
} /* namespace platform */
#endif /* GRIDDATA_H */