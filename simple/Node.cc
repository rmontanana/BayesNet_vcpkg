#include <string>
#include <vector>
#include <map>
#include "Node.h"

namespace bayesnet {
    Node::Node(std::string name) : name(name) {}

    void Node::addParent(Node* parent)
    {
        parents.push_back(parent);
        parent->children.push_back(this);
    }
}
