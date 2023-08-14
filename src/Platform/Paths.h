#ifndef PATHS_H
#define PATHS_H
namespace platform {
    class Paths {
    public:
        static std::string datasets() { return "datasets/"; }
        static std::string results() { return "results/"; }
    };
}
#endif