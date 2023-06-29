#ifndef NODE_H
#define NODE_H
#include <string>
#include <vector>
#include <map>
namespace bayesnet {
    class Node {
    private:
        std::string name;
        std::vector<Node*> parents;
        std::vector<Node*> children;
        std::map<std::vector<bool>, double> cpt; // Conditional Probability Table
    public:
        Node(std::string);
        void addParent(Node*);
    };
}
#endif
