#ifndef PATHS_H
#define PATHS_H
#include <string>
namespace platform {
    class Paths {
    public:
        static std::string datasets() { return "datasets/"; }
        static std::string results() { return "results/"; }
        static std::string excel() { return "excel/"; }
    };
}
#endif