#ifndef BESTRESULT_H
#define BESTRESULT_H
#include <string>
class BestResult {
public:
    static std::string title() { return "STree_default (linear-ovo)"; }
    static double score() { return 22.109799; }
    static std::string scoreName() { return "accuracy"; }
};
#endif