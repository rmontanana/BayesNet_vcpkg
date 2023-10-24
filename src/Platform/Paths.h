#ifndef PATHS_H
#define PATHS_H
#include <string>
#include "DotEnv.h"
namespace platform {
    class Paths {
    public:
        static std::string results() { return "results/"; }
        static std::string hiddenResults() { return "hidden_results/"; }
        static std::string excel() { return "excel/"; }
        static std::string cfs() { return "cfs/"; }
        static std::string datasets()
        {
            auto env = platform::DotEnv();
            return env.get("source_data");
        }
        static std::string excelResults() { return "some_results.xlsx"; }
    };
}
#endif