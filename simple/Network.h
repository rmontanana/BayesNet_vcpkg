#ifndef NETWORK_H
#define NETWORK_H
#include <string>
#include <vector>
#include <map>
#include "Node.h"

namespace bayesnet {
    class Network {
    private:
        std::map<std::string, Node*> nodes;
    public:
        Network();
        ~Network();
        void addNode(std::string);
        void addEdge(std::string, std::string);
        void fit(const std::vector<std::vector<double>>&);
        std::vector<double> predict(const std::vector<std::vector<double>>&);
    };
}
#endif
